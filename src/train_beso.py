import argparse
import os
import pathlib

_HF_DATASETS_CACHE = pathlib.Path(__file__).resolve().parents[1] / ".hf_datasets_cache"
_HF_DATASETS_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = str(_HF_DATASETS_CACHE)

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import NormalizationMode
from lerobot.policies import factory
from lerobot.scripts.lerobot_train import train as lerobot_train
from lerobot.utils.utils import init_logging
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from policies.beso.beso_config import BesoConfig

def train(data_dir="data"):
    print("\nStarting training...")
    dataset_cfg = DatasetConfig(
        repo_id=pathlib.Path(data_dir).name,
        root=data_dir,
        video_backend="torchcodec",
    )

    policy_overrides = {
        # Experiment overrides on top of BesoConfig defaults.
        "window_size": 4,
        "goal_conditioned": False,
        "goal_feature": "observation.goal.tail_q202",
        "goal_seq_len": 1,
        "use_amp": True,
        "freeze_rgb_encoder": True,
        "drop_n_last_frames": 0,
        "normalization_mapping": {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MIN_MAX,
        },
    }
    pretrained_config = BesoConfig(push_to_hub=False, **policy_overrides)
    print(f"[INFO] normalization_mapping={pretrained_config.normalization_mapping}")
    cfg = TrainPipelineConfig(
        policy=pretrained_config,
        dataset=dataset_cfg,
        batch_size=16,
        num_workers=8,
        steps=40000,
        save_freq=4000,
        log_freq=20,
        wandb=get_wandb_config(),
    )

    init_logging()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    force_cpu = pretrained_config.device == "cpu"
    mixed_precision = "bf16" if pretrained_config.use_amp and not force_cpu else "no"
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
        cpu=force_cpu,
        mixed_precision=mixed_precision,
    )
    lerobot_train(cfg, accelerator=accelerator)


def get_beso(_typename: str, **_kwargs):
    from policies.beso.modelling_beso import BesoPolicy

    return BesoPolicy


def get_wandb_config():
    return WandBConfig(
        enable=True,
        project="beso_lerobot",
        mode="online",
    )


def main():
    factory.get_policy_class = get_beso
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the dataset directory"
    )
    args = parser.parse_args()
    train(data_dir=pathlib.Path(args.data_dir))


if __name__ == "__main__":
    main()
