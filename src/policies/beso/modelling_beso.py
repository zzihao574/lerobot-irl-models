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
    populate_queues,
)
from lerobot.processor.normalize_processor import (
    NormalizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
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
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to load_state_dict before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.normalize_inputs = NormalizerProcessorStep(
            config.input_features, config.normalization_mapping, dataset_stats
        )
        self.normalize_targets = NormalizerProcessorStep(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = UnnormalizerProcessorStep(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_inputs = UnnormalizerProcessorStep(
            config.input_features, config.normalization_mapping, dataset_stats
        )
        self.step_counter = 0
        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None
        # self.
        self.diffusion = BesoModel(config)
        self._ema_helper = None
        self._ema_updates = 0
        self._ema_applied_for_eval = False
        if self.config.use_ema:
            self._reset_ema_from_current_weights()

    def get_optim_params(self) -> dict:
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
        if self._ema_helper is None:
            return
        self._ema_updates += 1
        if self._ema_updates % self.config.ema_update_every_n_steps == 0:
            self._ema_helper.update(self.diffusion.parameters())

    def _save_pretrained(self, save_directory):
        # Save EMA weights for checkpoints to match BESO source behavior.
        if self._ema_helper is None or self._ema_applied_for_eval:
            return super()._save_pretrained(save_directory)
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
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }

        if self.config.image_features:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(
                maxlen=self.config.n_obs_steps
            )
    
    # ========= inference  ============
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        queued_batch = {
            key: torch.stack(list(self._queues[key]), dim=1)
            for key in batch
            if key in self._queues
        }
        for key, value in batch.items():
            if key not in queued_batch:
                queued_batch[key] = value

        actions = self.diffusion.generate_actions(queued_batch)
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        # in select_action(...)

        if ACTION in batch:
            batch.pop(ACTION)
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(
                batch
            )  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        
        self._append_obs_queues(batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        self.step_counter += 1
        return action

    # ========= training  ============
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(
                batch
            )  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )

        batch = self.normalize_targets(batch)
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
        self.act_seq_len = config.horizon
        self.sampling_steps = config.sampling_steps
        self.goal_conditioned = config.goal_conditioned
        self.goal_feature = config.goal_feature
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

        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        if self.goal_conditioned:
            # Goal is represented as [B, G, D_goal] at runtime; use the last feature dim as D_goal.
            goal_shape = self.config.input_features[self.goal_feature].shape
            goal_dim = goal_shape[-1]

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
            goal_seq_len=self.config.goal_seq_len,
            obs_seq_len=self.config.n_obs_steps,
            action_seq_len=self.config.horizon,
            linear_output=self.config.linear_output,
            use_pos_emb=self.config.use_pos_emb,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            attn_pdrop=self.config.attn_pdrop,
            resid_pdrop=self.config.resid_pdrop,
            mlp_pdrop=self.config.mlp_pdrop,
            qk_norm=self.config.qk_norm,
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

    # ========= inference  ============
    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        goal_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        action_seq_len: int | None = None,
    ) -> Tensor:
        # Symbols:
        #   B=batch_size, T=horizon, A=action_dim, S=n_obs_steps, D_state=D_state_cond, G=goal_seq_len, D_goal=goal_dim
        # Inputs:
        #   global_cond: [B, S, D_state] or None
        #   goal_cond:   [B, G, D_goal] or None
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        if action_seq_len is None:
            action_seq_len = self.config.horizon
            
        # Sample Gaussian prior at sigma_max.
        actions = (
            torch.randn(
                size=(
                    batch_size,
                    action_seq_len,
                    self.config.action_feature.shape[0],
                ),
                dtype=dtype,
                device=device,
                generator=generator,
            )
            * self.sigma_max
        )
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
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        # 1) state as-is
        state_feats = batch[OBS_STATE]  # (B, S, state_dim)

        global_cond_feats = [state_feats]

        # 2) images -> encoder -> concat cameras
        img_features = None
        if self.config.image_features:
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
                    s=n_obs_steps,
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
                    s=n_obs_steps,
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
                text_features = text_outputs.unsqueeze(1).expand(-1, n_obs_steps, -1)
                global_cond_feats.append(text_features)

        feats = torch.cat(global_cond_feats, dim=-1)
        # feats / global_cond: [B, S, D_state_cond]

        return feats

    def _prepare_goal_conditioning(self, batch: dict[str, Tensor]) -> Tensor | None:
        if not self.goal_conditioned:
            return None

        # Clean goal interface: [B, G, D_goal]
        goal = batch[self.goal_feature]
        if goal.ndim != 3:
            raise ValueError(
                f"Goal tensor must have shape [B, G, D_goal], got {tuple(goal.shape)}"
            )
        if goal.shape[1] != self.config.goal_seq_len:
            raise ValueError(
                f"Goal sequence length mismatch: got {goal.shape[1]}, expected {self.config.goal_seq_len}"
            )
        # goal_cond: [B, G, D_goal]
        return goal

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert 1 <= n_obs_steps <= self.config.n_obs_steps, (
            f"Expected 1 <= n_obs_steps <= {self.config.n_obs_steps}, got {n_obs_steps}"
        )
        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, S, D_state_cond)
        goal_cond = self._prepare_goal_conditioning(batch)
        # Run DDIM sampling.
        actions = self.conditional_sample(
            batch_size,
            global_cond=global_cond,
            goal_cond=goal_cond,
            action_seq_len=n_obs_steps,
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
        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        global_cond = self._prepare_global_conditioning(batch)  # (B, S, D_state_cond)
        goal_cond = self._prepare_goal_conditioning(batch)

        # Forward diffusion.
        trajectory = batch["action"]  # [B, T, A]

        # Sample noise to add to the trajectory.
        noise = torch.randn(trajectory.shape, device=trajectory.device)  # [B, T, A]
        # Sample a random noising timestep for each item in the batch.
        device = trajectory.device

        sigmas = make_sample_density(
            self.config.sigma_sample_density_type,
            self.config.sigma_max,
            self.config.sigma_min,
        )(
            shape=(len(trajectory),),
            device=device,
        ).to(device)
        # sigmas: [B] (one sigma per sample, not the DDIM schedule)

        c_skip, c_out, c_in = [
            append_dims(x, trajectory.ndim) for x in self.get_scalings(sigmas)
        ]
        # c_skip, c_out, c_in: [B, 1, 1] (broadcast to [B, T, A])
        noised_input = trajectory + noise * append_dims(sigmas, trajectory.ndim)
        # noised_input: [B, T, A]
        model_output = self.dit_backbone(global_cond, noised_input * c_in, goal_cond, sigmas)
        # model_output: [B, T, A]
        target = (trajectory - c_skip * noised_input) / c_out
        # target: [B, T, A]
        
        loss = F.mse_loss(model_output, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
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
        return self.dit_backbone(state, action * c_in, goal, sigma, uncond=uncond) * c_out + action * c_skip

    # Preconditioned denoiser wrapper used by sample_ddim.
    def forward(self, state, action, goal, sigma, uncond: bool=False, cond_lambda: float | None = None):
        # state: [B, S, D_state_cond], action: [B, T, A], goal: [B, G, D_goal] or None, sigma: [B]
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
        dummy_shape_h_w = (
            config.crop_shape if config.crop_shape is not None else images_shape[1:]
        )
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
        # If we're going to crop, first up/downscale to 256x256 so the crop has enough context.
        if self.do_crop:
            # Bilinear resize on a batch tensor; preserves value range [0,1].
            x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

            if self.training:  # noqa: SIM108
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)

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
