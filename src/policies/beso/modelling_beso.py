from collections import deque
from collections.abc import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
)
from lerobot.processor.normalize_processor import (
    NormalizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from safetensors.torch import save_model as save_model_as_safetensor
from torch import Tensor, nn
from transformers import CLIPModel, CLIPProcessor

from .beso_config import BesoConfig
from .beso_transformer import Noise_Dec_only
from .utils import (
    ExponentialMovingAverage,
    append_dims,
    get_sigmas_exponential,
    make_sample_density,
    sample_ddim,
)


class BesoPolicy(PreTrainedPolicy):
    """
    Adapted from Lerobot implementation of Diffusion Policy.
    """

    config_class = BesoConfig
    name = "beso"

    def __init__(
        self,
        config: BesoConfig,
        dataset_meta,
        dataset_stats=None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_meta: LeRobot dataset metadata.
            dataset_stats: Optional explicit normalization stats override.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config
        norm_stats = dataset_stats if dataset_stats is not None else dataset_meta.stats
        self.normalize_inputs = NormalizerProcessorStep(
            config.input_features, config.normalization_mapping, norm_stats
        )
        self.normalize_targets = NormalizerProcessorStep(
            config.output_features, config.normalization_mapping, norm_stats
        )
        self.unnormalize_outputs = UnnormalizerProcessorStep(
            config.output_features, config.normalization_mapping, norm_stats
        )
        self.unnormalize_inputs = UnnormalizerProcessorStep(
            config.input_features, config.normalization_mapping, norm_stats
        )
        self.step_counter = 0
        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None
        self._action_context = None
        # self.
        self.diffusion = BesoModel(config)
        self._ema_helper = None
        self._ema_updates = 0
        self._ema_applied_for_eval = False
        self._action_clip_bounds = self._build_action_clip_bounds()

    def get_optim_params(self):
        """
        Return optimizer parameter groups with differential LR.
        """
        if (
            not self.config.freeze_rgb_encoder
            and hasattr(self.diffusion, "rgb_encoder")
        ):
            enc_params = list(self.diffusion.rgb_encoder.parameters())
            enc_ids = {id(p) for p in enc_params}
            other_params = [
                p for p in self.diffusion.parameters() if id(p) not in enc_ids
            ]
            return [
                {"params": enc_params, "lr": self.config.rgb_encoder_lr},
                {"params": other_params},
            ]
        return self.diffusion.parameters()

    def _store_original_weights_and_apply_ema(self):
        if self._ema_helper is None or self._ema_applied_for_eval:
            return
        self._ema_helper.store(self.diffusion.parameters())
        self._ema_helper.copy_to(self.diffusion.parameters())
        self._ema_applied_for_eval = True

    def _restore_original_weights(self):
        if self._ema_helper is None or not self._ema_applied_for_eval:
            return
        self._ema_helper.restore(self.diffusion.parameters())
        self._ema_applied_for_eval = False

    def _reset_ema_from_current_weights(self):
        self._ema_helper = ExponentialMovingAverage(
            self.diffusion.parameters(), self.config.ema_decay
        )
        self._ema_applied_for_eval = False

    def train(self, mode: bool = True):
        if mode:
            self._restore_original_weights()
        result = super().train(mode)
        if not mode:
            self._store_original_weights_and_apply_ema()
        return result

    def update(self):
        if not self.config.use_ema:
            return
        if self._ema_helper is None:
            self._reset_ema_from_current_weights()
            return
        self._ema_updates += 1
        if self._ema_updates % self.config.ema_update_every_n_steps == 0:
            self._ema_helper.update(self.diffusion.parameters())

    def _save_non_ema_weights(self, save_directory):
        model_to_save = self.module if hasattr(self, "module") else self
        save_model_as_safetensor(model_to_save, str(save_directory / "model_non_ema.safetensors"))

    def _save_pretrained(self, save_directory):
        # Save both non-EMA and EMA weights for checkpoints.
        if self._ema_helper is None or self._ema_applied_for_eval:
            return super()._save_pretrained(save_directory)
        self._save_non_ema_weights(save_directory)
        self._store_original_weights_and_apply_ema()
        try:
            return super()._save_pretrained(save_directory)
        finally:
            self._restore_original_weights()

    def load_state_dict(self, state_dict, strict: bool = True):
        incompatible = super().load_state_dict(state_dict, strict=strict)
        # Rebuild EMA shadow from the loaded weights. Otherwise a stale shadow from random init
        # can overwrite loaded parameters when `eval()` triggers EMA application.
        if self.config.use_ema:
            self._reset_ema_from_current_weights()
        return incompatible

    def _build_action_clip_bounds(self) -> tuple[Tensor, Tensor] | None:
        stats = self.normalize_targets._tensor_stats.get(ACTION)
        if not stats or "min" not in stats or "max" not in stats:
            return None

        action_feature = self.config.output_features.get(ACTION)
        if action_feature is None:
            return None

        min_norm = self.normalize_targets._apply_transform(
            stats["min"], ACTION, action_feature.type, inverse=False
        )
        max_norm = self.normalize_targets._apply_transform(
            stats["max"], ACTION, action_feature.type, inverse=False
        )
        return min_norm * 1.0, max_norm * 1.0

    def _clip_norm_actions(self, norm_actions: Tensor) -> Tensor:
        if self._action_clip_bounds is None:
            return torch.clamp(norm_actions, -1.1, 1.1)

        lower, upper = self._action_clip_bounds
        lower = lower.to(device=norm_actions.device, dtype=norm_actions.dtype)
        upper = upper.to(device=norm_actions.device, dtype=norm_actions.dtype)
        return torch.max(torch.min(norm_actions, upper), lower)

    def _append_obs_queues(self, batch: dict[str, Tensor]):
        """
        Append latest observation tensors to queues without cold-start padding.
        Unlike lerobot.populate_queues(...), this keeps the queue short at rollout start.
        """
        for key, q in self._queues.items():
            if key == ACTION:
                continue
            q.append(batch[key])

    def reset(self):
        """Clear observation and action queues. Should be called on env.reset()"""
        window_size = self.config.window_size
        self._queues = {
            "observation.state": deque(maxlen=window_size),
            "action": deque(maxlen=1),
        }
        self._action_context = deque(maxlen=window_size - 1)

        if self.config.image_features:
            self._queues["observation.images"] = deque(maxlen=window_size)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(
                maxlen=window_size
            )
    
    # ========= inference  ============
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Predict the latest action from queued observations.

        Returns:
            norm_actions: [B, 1, A] in normalized policy space (for action_context)
            env_actions: [B, 1, A] in environment action space (for execution)
        """
        queued_batch = {
            key: torch.stack(list(self._queues[key]), dim=1)
            for key in batch
            if key in self._queues
        }
        if len(self._action_context) > 0:
            queued_batch["action_context"] = torch.stack(list(self._action_context), dim=1)
        for key, value in batch.items():
            if key not in queued_batch:
                queued_batch[key] = value

        norm_actions = self.diffusion.generate_actions(queued_batch)
        norm_actions = self._clip_norm_actions(norm_actions)
        env_actions = self.unnormalize_outputs({ACTION: norm_actions})[ACTION]
        return norm_actions, env_actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        if ACTION in batch:
            batch.pop(ACTION)
        batch = dict(batch)  # shallow copy
        for key, feature in self.normalize_inputs.features.items():
            if key in batch:
                batch[key] = self.normalize_inputs._apply_transform(
                    batch[key], key, feature.type, inverse=False,
                )

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        
        self._append_obs_queues(batch)

        if len(self._queues[ACTION]) == 0:
            norm_actions, env_actions = self.predict_action_chunk(batch)
            self._action_context.extend(norm_actions.transpose(0, 1))
            self._queues[ACTION].extend(env_actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        self.step_counter += 1
        return action

    # ========= training  ============
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        # In lerobot_train, `preprocessor(batch)` already applies normalization for
        loss = self.diffusion.compute_loss(batch)
        # no output_dict so returning None
        return loss, None


class BesoModel(nn.Module):
    def __init__(self, config: BesoConfig):
        super().__init__()
        self.config = config
        self.sigma_data = config.sigma_data
        self.sigma_max = config.sigma_max
        self.sigma_min = config.sigma_min
        self.window_size = config.window_size
        self.sampling_steps = config.sampling_steps
        self.goal_conditioned = config.goal_conditioned
        self.goal_feature = config.goal_feature
        self.goal_seq_len = 0
        # Build observation encoders (depending on which observations are provided).
        global_cond_dim = self.config.robot_state_feature.shape[0]
        goal_dim = 0

        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [BesoRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = BesoRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images

            self._freeze_rgb_encoder_if_needed()

        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        if self.goal_conditioned:
            goal_shape = tuple(self.config.input_features[self.goal_feature].shape)
            if len(goal_shape) == 1:
                self.goal_seq_len = 1
                goal_dim = int(goal_shape[0])
            elif len(goal_shape) == 2:
                self.goal_seq_len = int(goal_shape[0])
                goal_dim = int(goal_shape[1])
            else:
                raise ValueError(
                    f"Unsupported goal feature shape for {self.goal_feature}: {goal_shape}."
                )

        # Initialize CLIP text encoder for language instructions
        if self.config.use_language:
            self.clip_model = CLIPModel.from_pretrained(self.config.clip_model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(
                self.config.clip_model_name
            )

            # Freeze CLIP if specified
            if self.config.freeze_clip:
                for param in self.clip_model.parameters():
                    param.requires_grad = False

            # Add CLIP text embedding dimension to global conditioning
            clip_text_dim = self.clip_model.config.text_config.hidden_size
            global_cond_dim += clip_text_dim

        self.dit_backbone = Noise_Dec_only(
            state_dim=global_cond_dim,
            action_dim=self.config.action_feature.shape[0],
            goal_dim=goal_dim,
            goal_conditioned=self.goal_conditioned,
            cond_mask_prob=self.config.cond_mask_prob,
            embed_dim=config.embed_dim,
            embed_pdrob=config.embed_pdrop,
            goal_seq_len=self.goal_seq_len,
            window_size=self.window_size,
            linear_output=self.config.linear_output,
            use_pos_emb=self.config.use_pos_emb,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            attn_pdrop=self.config.attn_pdrop,
            resid_pdrop=self.config.resid_pdrop,
            mlp_pdrop=self.config.mlp_pdrop,
            qk_norm=self.config.qk_norm,
            norm_type=self.config.norm_type,
            mlp_type=self.config.mlp_type,
            mlp_bias=self.config.mlp_bias,
            attention_impl=self.config.attention_impl,
        )
        self.device = None

        # Print parameter counts
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        print("BESO Model Parameter Count:")
        print("=" * 40)
        if hasattr(self, "rgb_encoder"):
            if isinstance(self.rgb_encoder, nn.ModuleList):
                total_encoder_params = sum(
                    count_params(enc) for enc in self.rgb_encoder
                )
                single_encoder_params = count_params(self.rgb_encoder[0])
                print(f"RGB Encoders (total): {total_encoder_params:,}")
                print(f"Single RGB Encoder: {single_encoder_params:,}")
            else:
                encoder_params = count_params(self.rgb_encoder)
                print(f"RGB Encoder: {encoder_params:,}")

        if hasattr(self, "clip_model"):
            clip_params = count_params(self.clip_model)
            print(f"CLIP Text Encoder: {clip_params:,}")

        backbone_params = count_params(self.dit_backbone)
        print(f"Transformer Backbone: {backbone_params:,}")

        total_params = count_params(self)
        print(f"Total Parameters: {total_params:,}")

        print("=" * 40)

    def _freeze_rgb_encoder_if_needed(self):
        if not self.config.freeze_rgb_encoder or not hasattr(self, "rgb_encoder"):
            return
        if isinstance(self.rgb_encoder, nn.ModuleList):
            for encoder in self.rgb_encoder:
                encoder.requires_grad_(False)
                encoder.eval()
        else:
            self.rgb_encoder.requires_grad_(False)
            self.rgb_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_rgb_encoder_if_needed()
        return self

    # ========= inference  ============
    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        goal_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        T_cur: int | None = None,
        action_context: Tensor | None = None,
    ) -> Tensor:
        # Symbols:
        #   B=batch_size, T_cur=current action sequence length, S_cur=current observation sequence length,
        #   A=action_dim, D_state=D_state_cond, G=goal_seq_len, D_goal=goal_dim
        # Inputs:
        #   global_cond: [B, S_cur, D_state] or None
        #   goal_cond:   [B, G, D_goal] or None
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        if T_cur is None:
            T_cur = self.window_size
            
        action_dim = self.config.action_feature.shape[0]
        if action_context is None:
            # Cold-start (aligned with upstream BESO): sample a single noisy action token.
            actions = (
                torch.randn(
                    size=(batch_size, 1, action_dim),
                    dtype=dtype,
                    device=device,
                    generator=generator,
                )
                * self.sigma_max
            )
        else:
            action_context = action_context.to(device=device, dtype=dtype)
            noise_last = (
                torch.randn(
                    size=(batch_size, 1, action_dim),
                    dtype=dtype,
                    device=device,
                    generator=generator,
                )
                * self.sigma_max
            )
            actions = torch.cat([action_context, noise_last], dim=1)
        # actions: [B, T, A]
        input_state = global_cond
        sigmas = get_sigmas_exponential(
            self.config.sampling_steps,
            self.config.sigma_min,
            self.config.sigma_max,
            device,
        )
        
        extra_args = None
        if self.goal_conditioned and goal_cond is not None:
            extra_args = {"cond_lambda": self.config.cond_lambda}

        actions = sample_ddim(self, input_state, actions, goal_cond, sigmas, extra_args=extra_args)
        # actions (denoised sample): [B, T, A]

        return actions

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, S_cur = batch[OBS_STATE].shape[:2]
        # 1) state as-is
        state_feats = batch[OBS_STATE]  # [B, S_cur, D_state]

        global_cond_feats = [state_feats]

        # 2) images -> encoder -> concat cameras
        img_features = None
        if self.config.image_features:
            with torch.no_grad() if self.config.freeze_rgb_encoder else torch.enable_grad():
                if self.config.use_separate_rgb_encoder_per_camera:
                    images_per_camera = einops.rearrange(
                        batch["observation.images"], "b s n ... -> n (b s) ..."
                    )
                    # images_per_camera: [N_cam, B*S, C, H, W]
                    img_features_list = torch.cat(
                        [
                            encoder(images)
                            for encoder, images in zip(
                                self.rgb_encoder, images_per_camera, strict=True
                            )
                        ]
                    )
                    img_features = einops.rearrange(
                        img_features_list,
                        "(n b s) ... -> b s (n ...)",
                        b=batch_size,
                        s=S_cur,
                    )
                    # img_features: [B, S, N_cam*D_img]
                else:
                    num_cameras = len(self.config.image_features)
                    shared = self.rgb_encoder(
                        einops.rearrange(
                            batch["observation.images"], "b s n ... -> (b s n) ..."
                        )
                    )
                    # shared: [B*S*N_cam, D_img]
                    img_features = einops.rearrange(
                        shared,
                        "(b s n) d -> b s (n d)",
                        b=batch_size,
                        s=S_cur,
                        n=num_cameras,
                    )
                    # img_features: [B, S, N_cam*D_img]
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        # 3) language instructions -> CLIP text encoder
        if self.config.use_language and self.config.language_feature in batch:
            # Get language instructions from batch
            # Expecting batch[language_feature] to be a list of strings
            language_instructions = batch[self.config.language_feature]

            # Handle both list and tensor inputs
            if isinstance(language_instructions, torch.Tensor):
                # If it's a tensor, assume it's token IDs and convert to text
                # For now, we'll skip this case and require string inputs
                raise ValueError(
                    "Language instructions must be provided as a list of strings"
                )

            # Process text with CLIP
            device = get_device_from_parameters(self)
            with torch.no_grad() if self.config.freeze_clip else torch.enable_grad():
                # Tokenize and encode text
                text_inputs = self.clip_processor(
                    text=language_instructions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(device)

                # Get text features from CLIP
                text_outputs = self.clip_model.get_text_features(**text_inputs)
                # text_outputs shape: (batch_size, clip_text_dim)

                # Expand to match observation steps and add to conditioning
                # Shape: (B, S, clip_text_dim)
                text_features = text_outputs.unsqueeze(1).expand(-1, S_cur, -1)
                global_cond_feats.append(text_features)

        feats = torch.cat(global_cond_feats, dim=-1)
        # feats / global_cond: [B, S_cur, D_state_cond]

        return feats

    def _prepare_goal_conditioning(self, batch):
        if not self.goal_conditioned:
            return None
        goal = batch[self.goal_feature]
        if goal.ndim == 4:
            goal = goal[:, -1]  # [B, G, D]
        elif goal.ndim == 2:
            goal = goal.unsqueeze(1)  # [B, 1, D]
        if goal.ndim != 3:
            raise ValueError(
                f"Unsupported goal tensor shape for {self.goal_feature}: {tuple(goal.shape)}"
            )
        if goal.shape[1] != self.goal_seq_len:
            raise ValueError(
                f"Goal sequence length mismatch: expected {self.goal_seq_len}, got {goal.shape[1]}."
            )
        return goal.float()

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, S_cur = batch["observation.state"].shape[:2]
        assert 1 <= S_cur <= self.window_size, (
            f"Expected 1 <= S_cur <= window_size ({self.window_size}), got {S_cur}"
        )
        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, S, D_state_cond)
        goal_cond = self._prepare_goal_conditioning(batch)
        action_context = batch.get("action_context")
        # Run DDIM sampling.
        actions = self.conditional_sample(
            batch_size,
            global_cond=global_cond,
            goal_cond=goal_cond,
            T_cur=S_cur,
            action_context=action_context,
        )
        # actions: [B, T, A]

        # With source-aligned interleaved BESO (T_cur == S_cur), the executable action is the last action token.
        # During cold start (short sequence) and after window is full, we return the latest action only.
        actions = actions[:, -1:, :]

        return actions
    
    # ========= training  ============
    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        # Input validation.
        assert set(batch).issuperset({"observation.state", "action", "action_is_pad"})
        assert "observation.images" in batch or "observation.environment_state" in batch
        S_cur = batch["observation.state"].shape[1]
        assert S_cur == self.window_size, (
            f"Training expects fixed window_size={self.window_size}, got S_cur={S_cur}"
        )
        global_cond = self._prepare_global_conditioning(batch)  # (B, S, D_state_cond)
        goal_cond = self._prepare_goal_conditioning(batch)

        # Forward diffusion.
        trajectory = batch["action"]  # [B, T, A]
        # Align train-time boundary behavior with eval cold-start: use zero-filled padded history.
        trajectory = trajectory.masked_fill(batch["action_is_pad"].unsqueeze(-1), 0.0)

        # Sample noise to add to the trajectory.
        noise = torch.randn(trajectory.shape, device=trajectory.device)  # [B, T, A]
        # Sample a random noising timestep for each item in the batch.
        device = trajectory.device

        sigmas = make_sample_density(
            self.config.sigma_sample_density_type,
            self.config.sigma_max,
            self.config.sigma_min,
            self.sigma_data,
        )(
            shape=(len(trajectory),),
            device=device,
        ).to(device)
        # sigmas: [B] (one sigma per sample, not the DDIM schedule)
        # Upstream BESO default: diffuse and supervise the full action sequence.
        c_skip, c_out, c_in = [
            append_dims(x, trajectory.ndim) for x in self.get_scalings(sigmas)
        ]
        noised_input = trajectory + noise * append_dims(sigmas, trajectory.ndim)
        model_output = self.dit_backbone(global_cond, noised_input * c_in, goal_cond, sigmas)
        target = (trajectory - c_skip * noised_input) / c_out
        loss = F.mse_loss(model_output, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            in_episode_bound = ~batch["action_is_pad"]  # [B, T]
            loss = loss * in_episode_bound.unsqueeze(-1)
        loss = loss.mean()
        return loss

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in
    
    def _forward_precond(self, state, action, goal, sigma, uncond: bool=False):
        c_skip, c_out, c_in = [
            append_dims(x, action.ndim) for x in self.get_scalings(sigma)
        ]
        net_out = self.dit_backbone(state, action * c_in, goal, sigma, uncond=uncond)
        return net_out * c_out + action * c_skip

    # Preconditioned denoiser wrapper used by sample_ddim.
    def forward(self, state, action, goal, sigma, uncond: bool=False, cond_lambda: float | None = None):
        # state: [B, S, D_state_cond], action: [B, T, A], goal: [B, G, D_goal] or None, sigma: [B]
        if cond_lambda is None:
            cond_lambda = self.config.cond_lambda
        # No goal path (or explicit unconditional branch): single forward
        if (not self.goal_conditioned) or (goal is None) or uncond or cond_lambda == 1.0:
            return self._forward_precond(state, action, goal, sigma, uncond=uncond)

        # CFG sampling (source-equivalent behavior):
        # out = out_uncond + cond_lambda * (out_cond - out_uncond)
        out_cond = self._forward_precond(state, action, goal, sigma, uncond=False)
        out_uncond = self._forward_precond(state, action, goal, sigma, uncond=True)
        return out_uncond + cond_lambda * (out_cond - out_uncond)

class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    (https://huggingface.co/papers/1509.06113). A minimal port of the robomimic implementation.

    At a high level, this takes 2D feature maps (from a convnet/ViT) and returns the "center of mass"
    of activations of each channel, i.e., keypoints in the image space for the policy to focus on.

    Example: take feature maps of size (512x10x12). We generate a grid of normalized coordinates (10x12x2):
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    This is achieved by applying channel-wise softmax over the activations (512x120) and computing the dot
    product with the coordinates (120x2) to get expected points of maximal activation (512x2).

    The example above results in 512 keypoints (corresponding to the 512 input channels). We can optionally
    provide num_kp != None to control the number of keypoints. This is achieved by a first applying a learnable
    linear mapping (in_channels, H, W) -> (num_kp, H, W).
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class BesoRgbEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, config: BesoConfig):
        super().__init__()
        # Set up optional preprocessing.
        if config.crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(
                    config.crop_shape
                )
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False
        # Optional resize (applied before crop, or alone when crop_shape=None)
        self.resize_shape = getattr(config, "resize_shape", None)
        # self.register_buffer(
        #     "img_mean",
        #     torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        # )
        # self.register_buffer(
        #     "img_std",
        #     torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        # )


        # Set up backbone.
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16, num_channels=x.num_features
                ),
            )

        # Note: we have a check in the config class to make sure all images have the same shape.
        images_shape = next(iter(config.image_features.values())).shape
        _resize = getattr(config, "resize_shape", None)
        if config.crop_shape is not None:
            dummy_shape_h_w = config.crop_shape
        elif _resize is not None:
            dummy_shape_h_w = _resize
        else:
            dummy_shape_h_w = images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]
        self.pool = SpatialSoftmax(
            feature_map_shape, num_kp=config.spatial_softmax_num_keypoints
        )
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Resize first (either to prepare for crop, or as sole preprocessing for full-image input).
        if self.resize_shape is not None:
            x = F.interpolate(x, size=self.resize_shape, mode="bilinear", align_corners=False)
        if self.do_crop:
            if self.training:  # noqa: SIM108
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)
        
        # # Normalize RGB to ImageNet stats before entering backbone.
        # x = (x - self.img_mean.to(device=x.device, dtype=x.dtype)) / self.img_std.to(
        #     device=x.device, dtype=x.dtype
        # )

        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer with non-linearity.
        x = self.relu(self.out(x))
        return x


def _replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    Args:
        root_module: The module for which the submodules need to be replaced
        predicate: Takes a module as an argument and must return True if the that module is to be replaced.
        func: Takes a module as an argument and returns a new module to replace it with.
    Returns:
        The root module with its submodules replaced.
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    assert not any(
        predicate(m) for _, m in root_module.named_modules(remove_duplicate=True)
    )
    return root_module
