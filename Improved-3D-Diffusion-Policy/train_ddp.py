"""
Usage:
Single node multi-GPU:
torchrun --standalone --nnodes=1 --nproc_per_node=4 train_ddp.py \
    --config-name=idp3.yaml \
    task=gr1_dex-3d
"""

import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict

import hydra
import torch
import torch.distributed as dist
from omegaconf import OmegaConf, open_dict
from termcolor import cprint
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
import diffusion_policy_3d.common.json_logger as json_logger_mod


# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

os.environ['WANDB_SILENT'] = "True"
OmegaConf.register_new_resolver("eval", eval, replace=True)


@dataclass
class DistContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


class AutoEpochDistributedSampler(DistributedSampler):
    """
    DistributedSampler that advances epoch automatically on each __iter__ call.
    This avoids touching every workspace loop to call sampler.set_epoch(epoch).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._auto_epoch = 0

    def __iter__(self):
        if self.shuffle:
            self.set_epoch(self._auto_epoch)
            self._auto_epoch += 1
        return super().__iter__()


class NullJsonLogger:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        return None

    def stop(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def log(self, data: Dict):
        return None

    def get_last_log(self):
        return None


def setup_distributed() -> DistContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1

    if enabled:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")

    return DistContext(
        enabled=enabled,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size
    )


def cleanup_distributed(ctx: DistContext):
    if ctx.enabled and dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def patch_dataloader_for_ddp(ctx: DistContext):
    if not ctx.enabled:
        return

    import torch.utils.data as tud
    original_dataloader = tud.DataLoader

    if getattr(original_dataloader, "_ddp_loader_patch", False):
        return

    class DDPDataLoader(original_dataloader):
        def __init__(self, dataset, *args, **kwargs):
            has_sampler = kwargs.get("sampler", None) is not None
            has_batch_sampler = kwargs.get("batch_sampler", None) is not None
            if not has_sampler and not has_batch_sampler:
                shuffle = bool(kwargs.get("shuffle", False))
                drop_last = bool(kwargs.get("drop_last", False))
                kwargs["sampler"] = AutoEpochDistributedSampler(
                    dataset=dataset,
                    num_replicas=ctx.world_size,
                    rank=ctx.rank,
                    shuffle=shuffle,
                    drop_last=drop_last
                )
                kwargs["shuffle"] = False
            super().__init__(dataset, *args, **kwargs)

    DDPDataLoader._ddp_loader_patch = True
    tud.DataLoader = DDPDataLoader


def patch_ema_for_ddp():
    if getattr(EMAModel.step, "_ddp_unwrap_patch", False):
        return

    original_step = EMAModel.step

    def step_with_ddp_unwrap(self, new_model):
        if isinstance(new_model, DDP):
            new_model = new_model.module
        return original_step(self, new_model)

    step_with_ddp_unwrap._ddp_unwrap_patch = True
    EMAModel.step = step_with_ddp_unwrap


def patch_checkpoint_for_ddp(ctx: DistContext):
    if getattr(BaseWorkspace.save_checkpoint, "_ddp_save_patch", False):
        return

    original_save_checkpoint = BaseWorkspace.save_checkpoint
    original_save_snapshot = BaseWorkspace.save_snapshot

    def _unwrap_ddp_modules(workspace: BaseWorkspace):
        replaced = dict()
        for key, value in workspace.__dict__.items():
            if isinstance(value, DDP):
                replaced[key] = value
                workspace.__dict__[key] = value.module
        return replaced

    def _rewrap_ddp_modules(workspace: BaseWorkspace, replaced: Dict):
        for key, value in replaced.items():
            workspace.__dict__[key] = value

    def save_checkpoint_rank0(self, *args, **kwargs):
        if ctx.enabled and not ctx.is_main:
            return None
        replaced = _unwrap_ddp_modules(self)
        try:
            return original_save_checkpoint(self, *args, **kwargs)
        finally:
            _rewrap_ddp_modules(self, replaced)

    def save_snapshot_rank0(self, *args, **kwargs):
        if ctx.enabled and not ctx.is_main:
            return None
        replaced = _unwrap_ddp_modules(self)
        try:
            return original_save_snapshot(self, *args, **kwargs)
        finally:
            _rewrap_ddp_modules(self, replaced)

    save_checkpoint_rank0._ddp_save_patch = True
    BaseWorkspace.save_checkpoint = save_checkpoint_rank0
    BaseWorkspace.save_snapshot = save_snapshot_rank0


def patch_logging_for_ddp(ctx: DistContext):
    if ctx.enabled and not ctx.is_main:
        json_logger_mod.JsonLogger = NullJsonLogger


def patch_workspace_model_ddp(workspace: BaseWorkspace, ctx: DistContext):
    if not ctx.enabled:
        return
    if not hasattr(workspace, "model"):
        return
    if not isinstance(workspace.model, torch.nn.Module):
        return

    find_unused = os.environ.get("DDP_FIND_UNUSED_PARAMETERS", "0").strip() == "1"
    broadcast_buffers = os.environ.get("DDP_BROADCAST_BUFFERS", "0").strip() == "1"
    model = workspace.model
    original_to = model.to
    wrapped = {"done": False}

    def to_and_wrap(*args, **kwargs):
        result = original_to(*args, **kwargs)
        if wrapped["done"]:
            return result

        if torch.cuda.is_available():
            ddp_model = DDP(
                model,
                device_ids=[ctx.local_rank],
                output_device=ctx.local_rank,
                find_unused_parameters=find_unused,
                broadcast_buffers=broadcast_buffers
            )
        else:
            ddp_model = DDP(
                model,
                find_unused_parameters=find_unused,
                broadcast_buffers=broadcast_buffers
            )

        workspace.model = ddp_model
        wrapped["done"] = True
        return ddp_model

    model.to = to_and_wrap


def patch_cfg_for_rank(cfg: OmegaConf, ctx: DistContext):
    with open_dict(cfg):
        if "training" in cfg:
            if torch.cuda.is_available():
                cfg.training.device = f"cuda:{ctx.local_rank}"
            else:
                cfg.training.device = "cpu"

        if ctx.enabled and not ctx.is_main:
            if "logging" in cfg and "mode" in cfg.logging:
                cfg.logging.mode = "disabled"
            if "checkpoint" in cfg and "save_ckpt" in cfg.checkpoint:
                cfg.checkpoint.save_ckpt = False


@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers use the same time
    OmegaConf.resolve(cfg)
    ctx = setup_distributed()

    try:
        patch_cfg_for_rank(cfg, ctx)
        patch_dataloader_for_ddp(ctx)
        patch_ema_for_ddp()
        patch_checkpoint_for_ddp(ctx)
        patch_logging_for_ddp(ctx)

        if ctx.enabled and ctx.is_main:
            cprint(
                f"[DDP] world_size={ctx.world_size}, backend={'nccl' if torch.cuda.is_available() else 'gloo'}",
                "cyan"
            )
        elif ctx.enabled and not ctx.is_main:
            print(
                f"[DDP] rank={ctx.rank}/{ctx.world_size}, local_rank={ctx.local_rank}, "
                f"device={cfg.training.device}"
            )

        cls = hydra.utils.get_class(cfg._target_)
        workspace: BaseWorkspace = cls(cfg)
        patch_workspace_model_ddp(workspace, ctx)
        workspace.run()
    finally:
        cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
