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
    default_kwargs = {
        # LeRobot vision encoder settings
        "vision_backbone": "resnet34",
        "pretrained_backbone_weights": "ResNet34_Weights.IMAGENET1K_V1",
        "use_group_norm": False,
        "crop_shape": (224, 224),
        "use_separate_rgb_encoder_per_camera": True,
        "down_dims": (128, 256),
        "kernel_size": 3,
        "n_groups": 8,
        "num_train_timesteps": 1000,
        "diffusion_step_embed_dim": 512,
        "prediction_type": "sample",
        # BESO algorithm settings (kept in BesoConfig, passed as kwargs here)
        "window_size": 4,  # maps to both n_obs_steps and horizon for interleaved BESO
        "n_action_steps": 1,
        "sampling_steps": 3,
        "sigma_min": 0.005,
        "sigma_max": 1.0,
        "use_ema": True,
        "ema_decay": 0.999,
        "ema_update_every_n_steps": 1,
        "linear_output": True,
        "spatial_softmax_num_keypoints": 32,
        "goal_conditioned": False,
        "optimizer_betas": (0.9, 0.999),
        "scheduler_warmup_steps": 100,   
    }
    pretrained_config = BesoConfig(push_to_hub=False, **default_kwargs)
    cfg = TrainPipelineConfig(
        policy=pretrained_config,
        dataset=dataset_cfg,
        batch_size=16,
        num_workers=0,
        steps=40000,
        save_freq=4000,
        log_freq=1,
        wandb=get_wandb_config(),
    )

    init_logging()
    lerobot_train(cfg)


def get_beso(typename: str, **kwargs):
    from policies.beso.modelling_beso import BesoPolicy

    return BesoPolicy


def get_wandb_config():
    wandb_config = WandBConfig(
        enable=True,
        project="beso_lerobot",
        mode="online",
    )
    return wandb_config


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
