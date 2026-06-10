if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)# Get repository root.
    sys.path.append(ROOT_DIR)# Add root to sys.path so other modules can be imported.
    os.chdir(ROOT_DIR)# Change working directory to root so relative paths resolve from there.

import os
import json
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import time
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
from dexpie.workspace.base_workspace import BaseWorkspace
from dexpie.workspace.critic_workspace import CriticWorkspace
from dexpie.policy.RTC_Recap import RTCRecapPolicy
from dexpie.policy.value_critic import ValueCritic
from dexpie.dataset.base_dataset import BaseImageDataset
from dexpie.common.checkpoint_util import TopKCheckpointManager
from dexpie.common.json_logger import JsonLogger
from dexpie.common.pytorch_util import dict_apply, optimizer_to
from dexpie.model.diffusion.ema_model import EMAModel
from dexpie.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class RecapWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    exclude_keys = ('value_critic',) # Exclude value_critic weights because they are trained in CriticWorkspace.

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: RTCRecapPolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: RTCRecapPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0
        self.ratio = cfg.training.ratio
        self.value_critic: ValueCritic = None
        self.adv_q_low = None
        self.adv_q_high = None
        self.advantage_quantiles_path = None

    def _load_frozen_value_critic(self, cfg: OmegaConf, device: torch.device):
        ckpt_path = cfg.get('value_critic_ckpt_path', None)
        if ckpt_path is None:
            raise ValueError("Missing `value_critic_ckpt_path` in config. "
                             "Please provide critic checkpoint path for Recap training.")
        ckpt_path = pathlib.Path(ckpt_path).expanduser()
        if not ckpt_path.is_file():
            raise ValueError(f"Value critic checkpoint file not found: {ckpt_path}")

        critic_workspace: CriticWorkspace = CriticWorkspace.create_from_checkpoint(str(ckpt_path))
        use_ema = bool(critic_workspace.cfg.training.use_ema)
        value_critic = critic_workspace.ema_model if (use_ema and critic_workspace.ema_model is not None) else critic_workspace.model
        value_critic.to(device)
        value_critic.eval()
        for p in value_critic.parameters():
            p.requires_grad = False

        self.value_critic = value_critic
        critic_cfg = critic_workspace.cfg
        self.max_length = critic_cfg.get('max_length', critic_cfg.task.dataset.max_length)
        cprint(f"[ValueCritic] loaded and frozen from: {ckpt_path}", "cyan")

    def _extract_quantile_threshold(self, stats_payload, target_q: float):
        quantile_list = stats_payload.get("advantage_quantiles", [])
        for item in quantile_list:
            if not isinstance(item, dict):
                continue
            q = item.get("quantile", None)
            thr = item.get("advantage_threshold", None)
            if q is None or thr is None:
                continue
            if abs(float(q) - float(target_q)) < 1e-6:
                return float(thr)

        key = f"q{target_q:.1f}"
        if key in stats_payload:
            return float(stats_payload[key])

        available_q = []
        for item in quantile_list:
            if isinstance(item, dict) and ("quantile" in item):
                try:
                    available_q.append(float(item["quantile"]))
                except Exception:
                    pass
        raise ValueError(
            f"Cannot find quantile {target_q:.3f} in {self.advantage_quantiles_path}. "
            f"Available quantiles: {available_q}"
        )

    def _load_advantage_quantiles(self, cfg: OmegaConf):
        stats_path = cfg.get("advantage_quantiles_json_path", None)
        if stats_path is None:
            raise ValueError(
                "Missing `advantage_quantiles_json_path` in config. "
                "Please provide advantage quantile json path for Recap training."
            )
        stats_path = pathlib.Path(hydra.utils.to_absolute_path(str(stats_path))).expanduser()
        if not stats_path.is_file():
            raise ValueError(f"Advantage quantiles json file not found: {stats_path}")

        self.advantage_quantiles_path = str(stats_path)

        quantile_low = float(cfg.get("advantage_quantile_low", 0.6))
        quantile_high = float(cfg.get("advantage_quantile_high", 0.8))

        with stats_path.open("r", encoding="utf-8") as f:
            stats_payload = json.load(f)

        q_low = self._extract_quantile_threshold(stats_payload, quantile_low)
        q_high = self._extract_quantile_threshold(stats_payload, quantile_high)

        if q_high <= q_low:
            raise ValueError(
                f"Invalid quantile thresholds in {stats_path}: "
                f"q{quantile_high:.1f}={q_high} must be larger than q{quantile_low:.1f}={q_low}."
            )

        self.adv_q_low = q_low
        self.adv_q_high = q_high

        cprint(
            (
                "[AdvQuant] loaded from: "
                f"{stats_path}, q{quantile_low:.1f}={q_low:.6f}, "
                f"q{quantile_high:.1f}={q_high:.6f}"
            ),
            "cyan",
        )

    def get_positive_mask(self, advantage: torch.Tensor):
        if self.adv_q_low is None:
            raise RuntimeError("Advantage quantiles are not loaded. Call `_load_advantage_quantiles` first.")
        quantile_thr = torch.as_tensor(self.adv_q_low, device=advantage.device, dtype=advantage.dtype)
        is_positive = advantage > quantile_thr
        return is_positive, quantile_thr

    def _prepare_obs_once(self, obs_dict):
        # Prepare image/wrist image once per batch for both value critic and recap loss.
        nobs = obs_dict.copy()
        image = nobs['image'] / 255.0
        if image.shape[-1] == 3:
            if len(image.shape) == 5:
                image = image.permute(0, 1, 4, 2, 3)
            if len(image.shape) == 4:
                image = image.permute(0, 3, 1, 2)
        nobs['image'] = image

        if "wrist_img" in nobs:
            wrist_img = nobs["wrist_img"] / 255.0
            if wrist_img.shape[-1] == 3:
                if len(wrist_img.shape) == 5:
                    wrist_img = wrist_img.permute(0, 1, 4, 2, 3)
                if len(wrist_img.shape) == 4:
                    wrist_img = wrist_img.permute(0, 3, 1, 2)
            nobs["wrist_img"] = wrist_img
        return nobs

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        # dataset element: {'obs', 'action'}
        # obs: {'image': (16,3,96,96) with range [0,1],  'agent_pos': (16,2)}
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer) # The training normalizer is an nn.Module, so deploy loads and uses it too.
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer) 

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

  
        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        self._load_frozen_value_critic(cfg, device)
        self._load_advantage_quantiles(cfg)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            verbose = True
        else:
            verbose = False
        
        
        RUN_VALIDATION = False # reduce time cost
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in tqdm.tqdm(range(cfg.training.num_epochs)):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                for batch_idx, batch in enumerate(train_dataloader):
                    # device transfer
                    t1 = time.time()
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    obs_for_loss = self._prepare_obs_once(batch['obs'])
                    batch_for_loss = batch.copy()
                    batch_for_loss['obs'] = obs_for_loss
                    
                    batch_value = self.value_critic(obs_for_loss, obs_preprocessed=True)  # [B,T,1]
                    valid_steps = batch["mask"].float().sum(dim=-1).clamp(min=1.0, max=float(batch_value.shape[1]))  # [B]
                    valid_offset = valid_steps - 1.0
                    mid_return = -(valid_offset / self.max_length)  # [B]
                    v_last = batch_value[:, -1, 0]   # [B]
                    v_first = batch_value[:, 0, 0]   # [B]
                    advantage = mid_return + v_last - v_first  # [B], V_{t+valid}-V_t
                    is_positive, quantile_thr = self.get_positive_mask(advantage)
                    adv_det = advantage.detach()
                    adv_mean = adv_det.mean().item()
                    adv_std = adv_det.std(unbiased=False).item()
                    adv_p90 = torch.quantile(adv_det, 0.9).item()
                    positive_ratio = is_positive.float().mean().item()
                    quantile_thr_value = quantile_thr.item()

                    raw_loss = self.model.compute_loss(batch_for_loss, is_positive=is_positive, obs_preprocessed=True)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    
                    # update ema
                    if cfg.training.use_ema:
                        ema.step(self.model)

                    # logging
                    raw_loss_cpu = raw_loss.item()
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'positive_ratio': positive_ratio,
                        'eps': quantile_thr_value,
                        'adv_mean': adv_mean,
                        'adv_std': adv_std,
                        'adv_p90': adv_p90,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }
                    
                    t2 = time.time()
                    if verbose:
                        print(f"total one step time: {t2-t1:.3f}")

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        wandb_run.log(step_log, step=self.global_step)
                        json_logger.log(step_log)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

              
                # run validation
                if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                    with torch.no_grad():
                        val_losses = list()
                    
                        for batch_idx, batch in enumerate(val_dataloader):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            obs_for_loss = self._prepare_obs_once(batch['obs'])
                            batch_for_loss = batch.copy()
                            batch_for_loss['obs'] = obs_for_loss
                            batch_value = self.value_critic(obs_for_loss, obs_preprocessed=True)  # [B,T,1]
                            valid_steps = batch["mask"].float().sum(dim=-1).clamp(min=1.0, max=float(batch_value.shape[1]))  # [B]
                            valid_offset = valid_steps - 1.0
                            mid_return = -(valid_offset / self.max_length)  # [B]
                            v_last = batch_value[:, -1, 0]   # [B]
                            v_first = batch_value[:, 0, 0]   # [B]
                            advantage = mid_return + v_last - v_first  # [B], V_{t+valid}-V_t
                            is_positive, _ = self.get_positive_mask(advantage)
                            loss = self.model.compute_loss(batch_for_loss, is_positive=is_positive, obs_preprocessed=True)
                            val_losses.append(loss)
                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
              
                step_log['test_mean_score'] = - train_loss
                    
                    
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        # stop wandb run
        wandb_run.finish()
    def eval(self):
        # load the latest checkpoint
        cfg = copy.deepcopy(self.cfg)
        
        lastest_ckpt_path = self.get_checkpoint_path()
        if lastest_ckpt_path.is_file():
            cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=lastest_ckpt_path)
        

        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.cuda()
        # runner_log = env_runner.run(policy)
        # cprint(f"---------------- Eval Results --------------", 'magenta')
        # for key, value in runner_log.items():
        #     if isinstance(value, float):
        #         cprint(f"{key}: {value:.4f}", 'magenta')
        
    def to_jit(self):
        # load the latest checkpoint
        
        cfg = copy.deepcopy(self.cfg)
        
        tag = "latest"
        # tag = "best"
        lastest_ckpt_path = self.get_checkpoint_path(tag=tag)
        
        if lastest_ckpt_path.is_file():
            cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=lastest_ckpt_path)
        lastest_ckpt_path = str(lastest_ckpt_path)
        jit_policy_path = lastest_ckpt_path.replace('.ckpt', '_jit.pt').replace("/checkpoints/", "/jit/")
        jit_policy_dir = os.path.dirname(jit_policy_path)
        os.makedirs(jit_policy_dir, exist_ok=True)

        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        device = torch.device('cpu')
        
        # configure dataset
        # dataset = hydra.utils.instantiate(cfg.task.dataset)
        # train_dataloader = DataLoader(dataset, **cfg.dataloader)
        # normalizer = dataset.get_normalizer()

        state_shape = cfg.task.shape_meta.obs.agent_pos.shape[0]
        img_shape = cfg.task.shape_meta.obs.image.shape
        dtype=policy.dtype
        with torch.no_grad():
            obs_dict = {
            'agent_pos': torch.ones((1, policy.n_obs_steps, state_shape), device=device, dtype=dtype),
            'image': torch.ones((1, policy.n_obs_steps, *img_shape), device=device, dtype=torch.int32),
            # 'image': torch.ones((1, policy.n_obs_steps, *img_shape), device=device),
            }
            result = policy(obs_dict)
            traced_policy = torch.jit.trace(policy, obs_dict)
            traced_policy.save(jit_policy_path)
        cprint(f"JIT policy saved to {jit_policy_path}", 'green')
        cprint("input shape for the policy:", 'yellow')
        cprint(f"agent_pos: {obs_dict['agent_pos'].shape}", 'yellow')
        cprint(f"image: {obs_dict['image'].shape}", 'yellow')

        cprint("-----------------------------", "yellow")

        # test the inference speed
        infer_times = 10
        t0 = time.time()
        for i in range(infer_times):
            with torch.no_grad():
                result = traced_policy(obs_dict)
        cprint(f"FPS for JIT policy: {infer_times/(time.time()-t0):.3f}", 'green')

        t0 = time.time()
        for i in range(infer_times):
            with torch.no_grad():
                result = policy(obs_dict)
        cprint(f"FPS for original policy: {infer_times/(time.time()-t0):.3f}", 'green')

        
    def get_model(self, ckpt_path=None):
        cfg = copy.deepcopy(self.cfg)
        
        if ckpt_path is None:
            tag = "latest"
            #tag = "best"
            lastest_ckpt_path = self.get_checkpoint_path(tag=tag)
            if lastest_ckpt_path.is_file():
                cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
                self.load_checkpoint(path=lastest_ckpt_path)
        else:
            if ckpt_path.is_file():
                cprint(f"Resuming from checkpoint {ckpt_path}", 'magenta')
                self.load_checkpoint(path=ckpt_path)
            else:
                raise ValueError(f"Checkpoint file not found: {ckpt_path}")

        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model    
        policy.eval()

        return policy
    
@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)

def main(cfg):
    workspace = RecapWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
