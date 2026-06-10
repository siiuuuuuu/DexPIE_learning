import json
import os
import pathlib
import random
import sys
from datetime import datetime
from typing import Dict, List, Tuple

if __name__ == "__main__":
    ROOT_DIR = pathlib.Path(__file__).resolve().parent
    sys.path.append(str(ROOT_DIR))
    os.chdir(str(ROOT_DIR))

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf, open_dict
from termcolor import cprint
from torch.utils.data import DataLoader

from dexpie.common.pytorch_util import dict_apply
from dexpie.workspace.critic_workspace import CriticWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


def resolve_device(cfg_device: str) -> torch.device:
    if cfg_device.startswith("cuda") and not torch.cuda.is_available():
        cprint(
            f"[Warn] cfg requires {cfg_device}, but CUDA is unavailable. Fallback to cpu.",
            "yellow",
        )
        return torch.device("cpu")
    return torch.device(cfg_device)


def prepare_obs_once(obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Keep exactly the same preprocessing as dexpie_workspace.
    nobs = obs_dict.copy()
    image = nobs["image"] / 255.0
    if image.shape[-1] == 3:
        if len(image.shape) == 5:
            image = image.permute(0, 1, 4, 2, 3)
        if len(image.shape) == 4:
            image = image.permute(0, 3, 1, 2)
    nobs["image"] = image

    if "wrist_img" in nobs:
        wrist_img = nobs["wrist_img"] / 255.0
        if wrist_img.shape[-1] == 3:
            if len(wrist_img.shape) == 5:
                wrist_img = wrist_img.permute(0, 1, 4, 2, 3)
            if len(wrist_img.shape) == 4:
                wrist_img = wrist_img.permute(0, 3, 1, 2)
        nobs["wrist_img"] = wrist_img
    return nobs


def load_frozen_value_critic(
    ckpt_path: pathlib.Path, device: torch.device
) -> Tuple[torch.nn.Module, float, bool]:
    critic_workspace: CriticWorkspace = CriticWorkspace.create_from_checkpoint(str(ckpt_path))
    use_ema = bool(critic_workspace.cfg.training.use_ema)
    value_critic = (
        critic_workspace.ema_model
        if (use_ema and critic_workspace.ema_model is not None)
        else critic_workspace.model
    )
    value_critic.to(device)
    value_critic.eval()
    for p in value_critic.parameters():
        p.requires_grad = False

    critic_cfg = critic_workspace.cfg
    max_length = critic_cfg.get("max_length", critic_cfg.task.dataset.max_length)
    return value_critic, float(max_length), use_ema


def parse_top_percentages(cfg: OmegaConf) -> List[float]:
    raw = cfg.get("advantage_top_percentages", [30, 40, 10, 50, 20])
    if isinstance(raw, (int, float)):
        raw = [raw]
    top_percentages = [float(x) for x in raw]
    for p in top_percentages:
        if not (0.0 < p < 100.0):
            raise ValueError(f"Top percentage must be in (0, 100), got: {p}")
    return top_percentages


@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.joinpath("dexpie", "config")),
    config_name="DexPIE",
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    seed = int(cfg.training.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    zarr_path = pathlib.Path(hydra.utils.to_absolute_path(str(cfg.task.dataset.zarr_path))).expanduser()
    with open_dict(cfg):
        cfg.task.dataset.zarr_path = str(zarr_path)

    ckpt_cfg_path = cfg.get("value_critic_ckpt_path", None)
    if ckpt_cfg_path is None:
        raise ValueError("Missing `value_critic_ckpt_path` in config.")
    ckpt_path = pathlib.Path(hydra.utils.to_absolute_path(str(ckpt_cfg_path))).expanduser()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Value critic checkpoint not found: {ckpt_path}")

    device = resolve_device(str(cfg.training.device))
    value_critic, max_length, use_ema = load_frozen_value_critic(ckpt_path, device)

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    dataloader_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)
    dataloader_cfg["shuffle"] = False
    dataloader = DataLoader(dataset, **dataloader_cfg)

    all_advantages: List[torch.Tensor] = []

    cprint(f"[Info] dataset size: {len(dataset)}", "cyan")
    cprint(f"[Info] critic ckpt: {ckpt_path}", "cyan")
    cprint(f"[Info] use EMA critic: {use_ema}", "cyan")
    cprint(f"[Info] max_length: {max_length}", "cyan")

    with torch.inference_mode():
        for batch in tqdm.tqdm(dataloader, desc="Collect advantages"):
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            obs_for_value = prepare_obs_once(batch["obs"])

            batch_value = value_critic(obs_for_value, obs_preprocessed=True)  # [B,T,1]
            valid_steps = batch["mask"].float().sum(dim=-1).clamp(min=1.0, max=float(batch_value.shape[1]))
            valid_offset = valid_steps - 1.0
            mid_return = -(valid_offset / max_length)
            v_last = batch_value[:, -1, 0]
            v_first = batch_value[:, 0, 0]
            advantage = mid_return + v_last - v_first  # [B]
            all_advantages.append(advantage.detach().cpu())

            del batch
            del obs_for_value
            del batch_value
            del v_last
            del v_first
            del advantage

    if len(all_advantages) == 0:
        raise RuntimeError("No advantages collected. Please check dataset/dataloader settings.")
    all_adv = torch.cat(all_advantages, dim=0).float()

    top_percentages = parse_top_percentages(cfg)
    top_stats = []
    for top_p in top_percentages:
        q = 1.0 - top_p / 100.0
        q_value = torch.quantile(all_adv, q).item()
        top_stats.append(
            {
                "top_percentage": top_p,
                "quantile": q,
                "advantage_threshold": q_value,
            }
        )

    zarr_out_dir = zarr_path if zarr_path.is_dir() else zarr_path.parent
    output_filename = str(cfg.get("advantage_quantile_filename", "advantage_quantiles.json"))
    output_path = zarr_out_dir / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_zarr_path": str(zarr_path),
        "value_critic_ckpt_path": str(ckpt_path),
        "critic_use_ema": bool(use_ema),
        "horizon": int(cfg.horizon),
        "max_length": max_length,
        "num_samples": int(all_adv.numel()),
        "advantage_quantiles": top_stats,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    cprint(f"[Done] Saved advantage quantiles to: {output_path}", "green")
    for item in top_stats:
        cprint(
            f"top {item['top_percentage']:.0f}% -> q={item['quantile']:.3f}, thr={item['advantage_threshold']:.6f}",
            "green",
        )


if __name__ == "__main__":
    main()
