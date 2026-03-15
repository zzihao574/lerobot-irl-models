import os
import sys
from pathlib import Path

import hydra
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies import factory
from lerobot.scripts.lerobot_train import train as lerobot_train
from lerobot.utils.utils import init_logging

from policies.beso.beso_config import BesoConfig


_HF_DATASETS_CACHE = Path(__file__).resolve().parents[1] / ".hf_datasets_cache"
_HF_DATASETS_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = str(_HF_DATASETS_CACHE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _build_accelerator(policy_cfg: BesoConfig) -> Accelerator:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    force_cpu = policy_cfg.device == "cpu"
    mixed_precision = "bf16" if policy_cfg.use_amp and not force_cpu else "no"
    return Accelerator(
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
        cpu=force_cpu,
        mixed_precision=mixed_precision,
    )

def get_beso(_typename: str, **_kwargs):
    from policies.beso.modelling_beso import BesoPolicy

    return BesoPolicy

@hydra.main(config_path="../configs/beso", config_name="train_beso", version_base="1.3")
def train(cfg) -> None:
    factory.get_policy_class = get_beso

    dataset_cfg = DatasetConfig(
        repo_id=cfg.repo_id,
        root=cfg.dataset_path,
        video_backend=cfg.video_backend,
    )

    init_logging()

    if cfg.train.mode == "resume":
        if not cfg.train.resume_config:
            raise ValueError("train.resume_config must be set when train.mode=resume")
        resume_config = Path(cfg.train.resume_config)
        resume_train_cfg = TrainPipelineConfig.from_pretrained(
            resume_config,
            cli_args=[
                "--resume=true",
                "--policy.load_non_ema=true",
                f"--dataset.root={Path(cfg.dataset_path)}",
                f"--dataset.repo_id={cfg.repo_id}",
            ],
        )

        if resume_train_cfg.policy is None:
            raise ValueError(f"No policy found in resume config: {resume_config}")

        accelerator = _build_accelerator(resume_train_cfg.policy)

        sys.argv = [
            sys.argv[0],
            f"--config_path={resume_config}",
        ]
        lerobot_train(resume_train_cfg, accelerator=accelerator)
        return

    policy_cfg = hydra.utils.instantiate(cfg.model, _convert_="all")

    if cfg.train.mode == "warmstart":
        if not cfg.train.pretrained_model_path:
            raise ValueError("train.pretrained_model_path must be set when train.mode=warmstart")
        pretrained_model_dir = Path(cfg.train.pretrained_model_path)
        policy_cfg = BesoConfig.from_pretrained(pretrained_model_dir)
        policy_cfg.pretrained_path = pretrained_model_dir
        policy_cfg.load_non_ema = False
        policy_cfg.push_to_hub = False

    train_cfg = TrainPipelineConfig(
        policy=policy_cfg,
        dataset=dataset_cfg,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        steps=cfg.train.steps,
        save_freq=cfg.train.save_freq,
        log_freq=cfg.train.log_freq,
        seed=cfg.train.seed,
        output_dir=Path(cfg.train.output_dir),
        job_name=cfg.train.job_name,
        wandb=WandBConfig(
            enable=cfg.wandb.enable,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            mode=cfg.wandb.mode,
        ),
    )

    accelerator = _build_accelerator(policy_cfg)
    lerobot_train(train_cfg, accelerator=accelerator)


if __name__ == "__main__":
    train()
