import argparse
import os
import pathlib

_HF_DATASETS_CACHE = pathlib.Path(__file__).resolve().parents[1] / ".hf_datasets_cache"
_HF_DATASETS_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = str(_HF_DATASETS_CACHE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import NormalizationMode
from lerobot.policies import factory
from lerobot.scripts.lerobot_train import train as lerobot_train
from lerobot.utils.utils import init_logging
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from policies.beso.beso_config import BesoConfig


def _parse_episodes(spec: str | None) -> list[int] | None:
    if spec is None or spec.strip() == "":
        return None
    episodes: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if start <= end else -1
            episodes.extend(range(start, end + step, step))
        else:
            episodes.append(int(token))
    return sorted(set(episodes))


def train(data_dir="data", episodes: list[int] | None = None):
    print("\nStarting training...")
    dataset_cfg = DatasetConfig(
        repo_id=pathlib.Path(data_dir).name,
        root=data_dir,
        episodes=episodes,
        video_backend="torchcodec",
    )

    policy_overrides = {
        # Experiment overrides on top of BesoConfig defaults.
        "window_size": 4,
        "goal_conditioned": False,
        "goal_feature": None,
        "goal_seq_len": 1,
        "use_amp": True,
        "freeze_rgb_encoder": False,
        "drop_n_last_frames": 0,
        # resize → random crop data augmentation
        "crop_shape": (384, 384),
        "crop_is_random": True,
        "resize_shape": (420, 420),
        # EDM noise schedule: sigma_max >> sigma_data so init is truly blind (SNR=0.01)
        "sigma_data": 1.0,
        "sigma_max": 4.0,
        "do_mask_loss_for_padding": True,

        "normalization_mapping": {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
    }
    pretrained_config = BesoConfig(push_to_hub=False, **policy_overrides)
    print(f"[INFO] normalization_mapping={pretrained_config.normalization_mapping}")
    print(f"[INFO] episodes={episodes}")
    cfg = TrainPipelineConfig(
        policy=pretrained_config,
        dataset=dataset_cfg,
        batch_size=24,
        num_workers=4,
        steps=30000,
        save_freq=2000,
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
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Subset episodes for training, e.g. '0,1' or '0-3,7'.",
    )
    args = parser.parse_args()
    train(
        data_dir=pathlib.Path(args.data_dir),
        episodes=_parse_episodes(args.episodes),
    )


if __name__ == "__main__":
    main()
