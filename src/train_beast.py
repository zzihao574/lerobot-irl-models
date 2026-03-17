import logging
import os
from pathlib import Path

import hydra
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies import factory
from lerobot.scripts.lerobot_train import train as lerobot_train
from lerobot.utils.utils import init_logging

from policies.beastf.beastf_config import BeastVLAConfig
from policies.beastf.modeling_beastf import BeastVLAPolicy

log = logging.getLogger(__name__)

def _build_accelerator(policy_cfg: BeastVLAConfig) -> Accelerator:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    force_cpu = policy_cfg.device == "cpu"
    mixed_precision = "bf16" if policy_cfg.use_amp and not force_cpu else "no"
    return Accelerator(
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
        cpu=force_cpu,
        mixed_precision=mixed_precision,
    )


@hydra.main(
    config_path="../configs/beast", config_name="train_beast", version_base="1.3"
)
def train(cfg):
    os.environ["LEROBOT_VIDEO_BACKEND"] = cfg.video_backend
    dataset_cfg = DatasetConfig(
        repo_id=cfg.repo_id,
        root=cfg.dataset_path,
        video_backend=cfg.video_backend,
    )
    pretrained_config = hydra.utils.instantiate(cfg.model, _convert_="all")
    pretrained_config.device = cfg.train.device
    pretrained_config.push_to_hub = cfg.train.push_to_hub
    pretrained_config.repo_id = cfg.repo_id

    train_cfg = TrainPipelineConfig(
        policy=pretrained_config,
        dataset=dataset_cfg,
        batch_size=cfg.train.batch_size,
        steps=cfg.train.steps,
        output_dir=Path(cfg.train.output_dir),
        job_name=cfg.train.job_name,
        save_freq=cfg.train.save_freq,
        seed=cfg.train.seed,
        log_freq=cfg.train.log_freq,
        num_workers=cfg.train.num_workers,
        wandb=WandBConfig(
        enable=cfg.wandb.enable,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        ),
	    )

    init_logging()
    accelerator = _build_accelerator(pretrained_config)
    lerobot_train(train_cfg, accelerator=accelerator)

def get_beast(typename: str, **kwargs):
    return BeastVLAPolicy

if __name__ == "__main__":
    factory.get_policy_class = get_beast
    train()
