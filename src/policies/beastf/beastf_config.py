from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("beast_vla")
@dataclass
class BeastVLAConfig(SmolVLAConfig):
    obs_modalities: str = "observation"
    goal_modalities: str = "task"
    target_modality: str = "action"
    task: str = ""
    lang_modalities: list[str] = field(default_factory=lambda: ["language_instruction"])
    img_modalities: list[str] = field(default_factory=lambda: ["observation.image.centric_cam"])

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    vlm_path: str = "microsoft/Florence-2-base"
    vlm_model_name: str = "bert-base-uncased"
    tokenizer_max_length: int = 77
    pad_language_to: str = "longest"
    freeze_florence: bool = False
    freeze_vision_tower: bool = False
    freeze_embeddings_only: bool = False
    token_dropout: float = 0.1
    cfg_dropout: float = 0.0
    cfg_lambda: float = 1.0
    prompt_robot_name: str = "Franka Panda"
    prompt_num_arms: int = 1
    prompt_action_space: str = "7 joint positions + 1 gripper width"
    prompt_include_meta: bool = True
    image_resize_hw: tuple[int, int] = (224, 224)
    image_use_clip_normalization: bool = True
    
    action_dim: int = 8
    act_window_size: int = 30
    chunk_size: int = 30
    n_action_steps: int = 30
    multistep: int = 30
    lowdim_obs_dim: int = 30

    use_second_view: bool = True
    second_view_key: str = "observation.image.wrist_cam"

    num_dof: int = 8
    gripper_dof: int = 1
    gripper_zero_order: bool = False
    enforce_init_pos: bool = True
    num_basis: int = 5
    degree_p: int = 4
    action_bins: int = 256
    update_w_bound: bool = True
    text_max_length: int = 77

    return_act_chunk: bool = False
    use_action_scale: bool = False
    use_early_cross_fusion: bool = True

    optimizer_lr: float = 2e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 400_000
    scheduler_decay_lr: float = 1e-5

    def __post_init__(self) -> None:
        self.chunk_size = self.act_window_size
        self.n_action_steps = self.multistep
        self.tokenizer_max_length = self.text_max_length
        self.pad_language_to = "longest"
        super().__post_init__()

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
        )
