import argparse
import os
import pathlib

_HF_DATASETS_CACHE = pathlib.Path(__file__).resolve().parents[1] / ".hf_datasets_cache"
_HF_DATASETS_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = str(_HF_DATASETS_CACHE)

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies import factory
from lerobot.scripts.lerobot_train import train as lerobot_train
from lerobot.utils.utils import init_logging

from policies.beso.beso_config import BesoConfig


def train(data_dir="data"):
    print("\nStarting training...")
    dataset_cfg = DatasetConfig(
        repo_id="banana_beso_clean_v1",
        root=data_dir,
        video_backend="torchcodec",
    )
    policy_overrides = {
        # Experiment overrides on top of BesoConfig defaults.
        "window_size": 4,
        "goal_conditioned": False,
        "goal_feature": "observation.goal.tail_q202",
        "goal_seq_len": 1,
    }
    pretrained_config = BesoConfig(push_to_hub=False, **policy_overrides)
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
    lerobot_train(cfg)


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
