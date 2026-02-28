from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


@PreTrainedConfig.register_subclass("beso")
class BesoConfig(DiffusionConfig):
    def __init__(
        self,
        # Sequence / rollout semantics (BESO interleaved transformer uses one shared window)
        window_size: int | None = None,
        n_action_steps: int = 1,  # LeRobot compatibility field; Scheme-B rollout executes one action per step.
        # BESO / EDM / DDIM hyperparameters
        sigma_data: float = 0.5,
        sigma_max: float = 1.0,
        sigma_min: float = 0.005,
        sampling_steps: int = 3,
        sigma_sample_density_type: str = "loglogistic",
        # Transformer / backbone parameters
        embed_dim: int = 360,
        n_layers: int = 6,
        n_heads: int = 6,
        attn_pdrop: float = 0.3,
        resid_pdrop: float = 0.0,
        mlp_pdrop: float = 0.0,
        embed_pdrop: float = 0.0,
        qk_norm: bool = False,
        norm_type: str = "layernorm", # rmsnorm or layernorm
        mlp_type: str = "gelu", # swishglu or gelu
        mlp_bias: bool = True,
        attention_impl: str = "auto",
        use_pos_emb: bool = True,
        linear_output: bool = True,
        # RGB encoder parameters (used directly by BesoRgbEncoder)
        vision_backbone: str = "resnet34",
        pretrained_backbone_weights: str | None = "ResNet34_Weights.IMAGENET1K_V1",
        use_group_norm: bool = False,
        crop_shape: tuple[int, int] | None = (192, 192),
        crop_is_random: bool = True,
        use_separate_rgb_encoder_per_camera: bool = False,
        freeze_rgb_encoder: bool = False,
        spatial_softmax_num_keypoints: int = 32,
        # EMA
        use_ema: bool = True,
        ema_decay: float = 0.999,
        ema_update_every_n_steps: int = 1,
        # Goal conditioning (kept optional; enable when dataset contains goal feature)
        goal_conditioned: bool = False,
        goal_feature: str | None = None,
        goal_seq_len: int = 1,
        cond_mask_prob: float = 0.1,
        cond_lambda: float = 1.5,  
        # Language instruction parameters
        use_language: bool = False,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        freeze_clip: bool = True,
        language_feature: str | None = None,
        # LeRobot compatibility knobs (inherited DiffusionConfig fields still used by presets/validation)
        down_dims: tuple[int, ...] = (128, 256),
        optimizer_betas: tuple = (0.9, 0.999),
        scheduler_warmup_steps: int = 100,
        **kwargs,
    ):
        kwargs["n_action_steps"] = n_action_steps
        kwargs["vision_backbone"] = vision_backbone
        kwargs["pretrained_backbone_weights"] = pretrained_backbone_weights
        kwargs["use_group_norm"] = use_group_norm
        kwargs["crop_shape"] = crop_shape
        kwargs["crop_is_random"] = crop_is_random
        kwargs["use_separate_rgb_encoder_per_camera"] = use_separate_rgb_encoder_per_camera
        kwargs["spatial_softmax_num_keypoints"] = spatial_softmax_num_keypoints
        kwargs["down_dims"] = down_dims
        kwargs["optimizer_betas"] = optimizer_betas
        kwargs["scheduler_warmup_steps"] = scheduler_warmup_steps

        if window_size is not None:
            kwargs["n_obs_steps"] = window_size
            kwargs["horizon"] = window_size

        super().__init__(**kwargs)

        # BESO interleaved transformer requires one state token per action token.
        if self.n_obs_steps != self.horizon:
            raise ValueError(
                "BESO interleaved transformer requires n_obs_steps == horizon. "
                f"Got n_obs_steps={self.n_obs_steps}, horizon={self.horizon}."
            )
        if self.n_action_steps != 1:
            raise ValueError(
                "Current BESO rollout (Scheme-B, source-aligned) executes one action per step, "
                f"so n_action_steps must be 1. Got {self.n_action_steps}."
            )
        if embed_dim % n_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads}).")
        if norm_type not in {"rmsnorm", "layernorm"}:
            raise ValueError(f"norm_type must be 'rmsnorm' or 'layernorm', got {norm_type!r}.")
        if mlp_type not in {"swishglu", "gelu"}:
            raise ValueError(f"mlp_type must be 'swishglu' or 'gelu', got {mlp_type!r}.")
        if attention_impl not in {"auto", "manual"}:
            raise ValueError(
                f"attention_impl must be 'auto' or 'manual', got {attention_impl!r}."
            )
        if goal_conditioned and goal_feature is None:
            raise ValueError("Set goal_feature when goal_conditioned=True.")
        if pretrained_backbone_weights is not None and use_group_norm:
            raise ValueError(
                "pretrained_backbone_weights and use_group_norm=True are incompatible for BesoRgbEncoder."
            )

        self.window_size = self.n_obs_steps

        # EDM-like scaling hyperparams
        self.embed_dim = embed_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sampling_steps = sampling_steps
        self.sigma_sample_density_type = sigma_sample_density_type

        # Backbone hyperparameters
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.mlp_pdrop = mlp_pdrop
        self.embed_pdrop = embed_pdrop
        self.qk_norm = qk_norm
        self.norm_type = norm_type
        self.mlp_type = mlp_type
        self.mlp_bias = mlp_bias
        self.attention_impl = attention_impl
        self.use_pos_emb = use_pos_emb
        self.linear_output = linear_output

        # RGB encoder parameters
        self.vision_backbone = vision_backbone
        self.pretrained_backbone_weights = pretrained_backbone_weights
        self.use_group_norm = use_group_norm
        self.crop_shape = crop_shape
        self.crop_is_random = crop_is_random
        self.use_separate_rgb_encoder_per_camera = use_separate_rgb_encoder_per_camera
        self.freeze_rgb_encoder = freeze_rgb_encoder
        self.spatial_softmax_num_keypoints = spatial_softmax_num_keypoints

        # EMA
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_update_every_n_steps = ema_update_every_n_steps

        # Goal conditioning
        self.goal_conditioned = goal_conditioned
        self.goal_feature = goal_feature
        self.goal_seq_len = goal_seq_len

        # CFG
        self.cond_mask_prob = cond_mask_prob
        self.cond_lambda = cond_lambda

        # Language parameters
        self.use_language = use_language
        self.clip_model_name = clip_model_name
        self.freeze_clip = freeze_clip
        self.language_feature = language_feature

        # LeRobot compatibility knobs (explicitly mirrored for clarity)
        self.down_dims = down_dims
        self.optimizer_betas = optimizer_betas
        self.scheduler_warmup_steps = scheduler_warmup_steps
