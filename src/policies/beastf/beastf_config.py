from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("beast_vla")
@dataclass
class BeastVLAConfig(PreTrainedConfig):
    # Modalities / keys.
    obs_modalities: str = "observation"
    goal_modalities: str = "task"
    target_modality: str = "action"
    lang_modalities: list[str] = field(default_factory=lambda: ["language_instruction"])
    img_modalities: list[str] = field(default_factory=lambda: ["observation.image.centric_cam"])

    # Define input and output features for normalization.
    input_features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "observation.image.centric_cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.image.wrist_cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        }
    )
    output_features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
        }
    )

    normalization_mapping: dict[FeatureType, NormalizationMode] = field(
        default_factory=lambda: {
            FeatureType.STATE: NormalizationMode.MEAN_STD,
            FeatureType.ACTION: NormalizationMode.MEAN_STD,
        }
    )

    # VLM configuration.
    vlm_path: str = "microsoft/Florence-2-base"
    freeze_florence: bool = False
    freeze_vision_tower: bool = False
    freeze_embeddings_only: bool = False
    vlm_prompt_style: str = "default"
    token_dropout: float = 0.1
    cfg_dropout: float = 0.0
    cfg_lambda: float = 1.0

    # Action / observation configuration.
    action_dim: int = 8
    act_window_size: int = 16
    chunk_size: int = 16
    multistep: int = 16
    lowdim_obs_dim: int = 16
    state_dim: int = 14
    use_proprio: bool = False

    # Image configuration.
    use_second_view: bool = True
    second_view_key: str = "observation.image.wrist_cam"

    # Beast Tokenizer configuration.
    num_dof: int = 8
    gripper_zero_order: bool = False
    num_basis: int = 5
    degree_p: int = 4
    action_bins: int = 256
    update_w_bound: bool = True

    # Action output configuration.
    return_act_chunk: bool = False

    # Additional features.
    use_action_scale: bool = False
    use_early_cross_fusion: bool = True

    @property
    def observation_delta_indices(self) -> list | None:  # type: ignore[override]
        return None

    @property
    def action_delta_indices(self) -> list | None:  # type: ignore[override]
        return None

    @property
    def reward_delta_indices(self) -> list | None:  # type: ignore[override]
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=2e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=1000,
            num_decay_steps=400_000,
            peak_lr=2e-5,
            decay_lr=1e-5,
        )

    def validate_features(self) -> None:
        # Keep this lightweight; the policy will raise KeyError if dataset keys mismatch.
        if "action" not in self.output_features:
            raise ValueError("BeastVLAConfig.output_features must contain 'action'.")
