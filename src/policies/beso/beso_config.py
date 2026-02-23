from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


@PreTrainedConfig.register_subclass("beso")
class BesoConfig(DiffusionConfig):
    def __init__(
        self,
        # BESO / EDM hyperparameters
        sigma_data: float = 0.5,
        sigma_max: float = 1.0,
        sigma_min: float = 0.005,
        sampling_steps: int = 3,
        sigma_sample_density_type: str = "loglogistic",
        # Sequence semantics (BESO interleaved transformer uses a single window length)
        window_size: int | None = None,
        # Transformer / backbone parameters
        embed_dim: int = 448,
        n_layers: int = 6,
        n_heads: int = 16,
        attn_pdrop: float = 0.3,
        resid_pdrop: float = 0.0,
        mlp_pdrop: float = 0.0,
        embed_pdrop: float = 0.0,
        qk_norm: bool = True,
        use_pos_emb: bool = True,
        linear_output: bool = True,
        # EMA
        use_ema: bool = True,
        ema_decay: float = 0.999,
        ema_update_every_n_steps: int = 1,
        # Goal conditioning (kept optional; enable when dataset contains goal feature)
        goal_conditioned: bool = False,
        goal_feature: str | None = None,
        goal_seq_len: int = 1,
        # Language instruction parameters
        use_language: bool = False,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        freeze_clip: bool = True,
        language_feature: str | None = None,
        **kwargs,
    ):
        if window_size is not None:
            if "n_obs_steps" in kwargs and kwargs["n_obs_steps"] != window_size:
                raise ValueError(
                    f"BESO window_size ({window_size}) conflicts with n_obs_steps ({kwargs['n_obs_steps']})."
                )
            if "horizon" in kwargs and kwargs["horizon"] != window_size:
                raise ValueError(
                    f"BESO window_size ({window_size}) conflicts with horizon ({kwargs['horizon']})."
                )
            kwargs["n_obs_steps"] = window_size
            kwargs["horizon"] = window_size

        super().__init__(**kwargs)

        # BESO interleaved transformer requires one state token per action token.
        if self.n_obs_steps != self.horizon:
            raise ValueError(
                "BESO interleaved transformer requires n_obs_steps == horizon. "
                f"Got n_obs_steps={self.n_obs_steps}, horizon={self.horizon}."
            )
        if self.n_action_steps > self.horizon:
            raise ValueError(
                f"n_action_steps must be <= horizon. Got {self.n_action_steps} > {self.horizon}."
            )
        if embed_dim % n_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads}).")
        if goal_conditioned and goal_feature is None:
            raise ValueError("Set goal_feature when goal_conditioned=True.")

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
        self.use_pos_emb = use_pos_emb
        self.linear_output = linear_output

        # EMA
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_update_every_n_steps = ema_update_every_n_steps

        # Goal conditioning
        self.goal_conditioned = goal_conditioned
        self.goal_feature = goal_feature
        self.goal_seq_len = goal_seq_len

        # Language parameters
        self.use_language = use_language
        self.clip_model_name = clip_model_name
        self.freeze_clip = freeze_clip
        self.language_feature = language_feature
