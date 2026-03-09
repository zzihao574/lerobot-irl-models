from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


@PreTrainedConfig.register_subclass("beso")
@dataclass
class BesoConfig(DiffusionConfig):
    # Sequence / rollout semantics.
    window_size: int | None = None
    n_action_steps: int = 1

    # BESO / EDM / DDIM hyperparameters.
    sigma_data: float = 0.5
    sigma_max: float = 1.0
    sigma_min: float = 0.005
    sampling_steps: int = 3
    sigma_sample_density_type: str = "loglogistic"

    # Transformer / backbone parameters.
    embed_dim: int = 360
    n_layers: int = 6
    n_heads: int = 6
    attn_pdrop: float = 0.2
    resid_pdrop: float = 0.1
    mlp_pdrop: float = 0.1
    embed_pdrop: float = 0.0
    qk_norm: bool = False
    norm_type: str = "layernorm"  # "rmsnorm" or "layernorm"
    mlp_type: str = "gelu"  # "swishglu" or "gelu"
    mlp_bias: bool = True
    attention_impl: str = "auto"  # "auto" or "manual"
    use_pos_emb: bool = True
    linear_output: bool = True

    # RGB encoder parameters.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    use_group_norm: bool = False
    crop_shape: tuple[int, int] | None = (192, 192)
    crop_is_random: bool = False
    resize_shape: tuple[int, int] | None = None
    use_separate_rgb_encoder_per_camera: bool = False
    freeze_rgb_encoder: bool = False
    spatial_softmax_num_keypoints: int = 32

    # EMA.
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_update_every_n_steps: int = 1

    # Goal conditioning.
    goal_conditioned: bool = False
    goal_feature: str | None = None

    # CFG.
    cond_mask_prob: float = 0.1
    cond_lambda: float = 1.5

    # Language instruction parameters.
    use_language: bool = False
    clip_model_name: str = "openai/clip-vit-base-patch32"
    freeze_clip: bool = True
    language_feature: str | None = None

    # Override DiffusionConfig defaults for BESO.
    down_dims: tuple[int, ...] = ()
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    scheduler_warmup_steps: int = 500

    # Differential LR for RGB encoder (only used when freeze_rgb_encoder=False).
    rgb_encoder_lr: float = 1e-5

    # Runtime load behavior.
    load_non_ema: bool = False

    def __post_init__(self) -> None:
        if self.window_size is not None:
            self.n_obs_steps = self.window_size
            self.horizon = self.window_size

        super().__post_init__()

        if self.n_obs_steps != self.horizon:
            raise ValueError(
                "BESO interleaved transformer requires n_obs_steps == horizon. "
                f"Got n_obs_steps={self.n_obs_steps}, horizon={self.horizon}."
            )

        if self.n_action_steps != 1:
            raise ValueError(
                "Current BESO rollout executes one action per step, "
                f"so n_action_steps must be 1. Got {self.n_action_steps}."
            )

        if self.embed_dim % self.n_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by n_heads ({self.n_heads})."
            )

        if self.norm_type not in {"rmsnorm", "layernorm"}:
            raise ValueError(
                f"norm_type must be 'rmsnorm' or 'layernorm', got {self.norm_type!r}."
            )

        if self.mlp_type not in {"swishglu", "gelu"}:
            raise ValueError(
                f"mlp_type must be 'swishglu' or 'gelu', got {self.mlp_type!r}."
            )

        if self.attention_impl not in {"auto", "manual"}:
            raise ValueError(
                f"attention_impl must be 'auto' or 'manual', got {self.attention_impl!r}."
            )

        if self.goal_conditioned and self.goal_feature is None:
            raise ValueError("Set goal_feature when goal_conditioned=True.")

        if self.pretrained_backbone_weights is not None and self.use_group_norm:
            raise ValueError(
                "pretrained_backbone_weights and use_group_norm=True are incompatible "
                "for BesoRgbEncoder."
            )

        self.window_size = self.n_obs_steps
