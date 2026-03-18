import logging
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from lerobot.policies.pretrained import PreTrainedPolicy
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig

from .beast_tokenizer.utils import discrete_to_continuous
from .beastf_config import BeastVLAConfig
# Assuming beast.py is in .beast_tokenizer package or similar
from .beast_tokenizer.beast import BeastTokenizer
from .beastf_utils import build_policy_prompt, create_bidirectional_mask, token_prediction_accuracy

from lerobot.processor.normalize_processor import UnnormalizerProcessorStep
from lerobot.utils.constants import ACTION

logger = logging.getLogger(__name__)

CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


class BeastVLAPolicy(PreTrainedPolicy):
    """
    BeastVLA Policy for LeRobot.
    """
    name = "beast_vla"
    config_class = BeastVLAConfig

    def __init__(
        self,
        config: BeastVLAConfig,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
        **kwargs,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.unnormalize_outputs = UnnormalizerProcessorStep(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.model = BeastFModel(config)
        self.model.reset()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        result = self.model.forward(batch)
        return result["loss"], result["loss_dict"]

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self.forward(batch)

    @torch.no_grad()
    def predict_action_chunk(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        cond = self.model.encode_observations(batch)
        norm_action_seq = self.model.sample_actions(None, cond, inference=True)
        env_action_seq = self.unnormalize_outputs({ACTION: norm_action_seq})[ACTION]
        return norm_action_seq, env_action_seq

    def reset(self) -> None:
        self.model.reset()

    @torch.no_grad()
    def select_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Select action for inference (LeRobot protocol).
        This method handles action chunking internally:
        - On first call (or every multistep): predicts new action chunk
        - Returns single action per timestep from the chunk

        Args:
            batch: Observation batch

        Returns:
            Selected action [B, action_dim]
        """
        if ACTION in batch:
            batch.pop(ACTION)
        # Check if we need to predict a new action chunk
        if (
            self.model.rollout_step_counter % self.config.multistep == 0
            or self.model.pred_action_seq_norm is None
            or self.model.pred_action_seq_env is None
        ):
            # Predict new action chunk
            (
                self.model.pred_action_seq_norm,
                self.model.pred_action_seq_env,
            ) = self.predict_action_chunk(batch)

        # Get current action from the chunk
        if self.config.return_act_chunk:
            # Return full chunk
            action = self.model.pred_action_seq_env
        else:
            # Return single action at current step
            action = self.model.pred_action_seq_env[:, self.model.rollout_step_counter, :]

        # Update counter
        self.model.rollout_step_counter += 1
        if self.model.rollout_step_counter >= self.config.multistep:
            self.model.rollout_step_counter = 0

        return action
    
    def get_optim_params(self) -> dict:
        return self.model.parameters()
    


class BeastFModel(nn.Module):
    def __init__(self, config: BeastVLAConfig):
        super().__init__()
        self.config = config
        self.task = config.task
        self.device = torch.device(
            config.device if hasattr(config, "device") and config.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # --- Init Helpers ---
        self._init_modalities(config)
        self._init_flags(config)

        # --- Setup VLM ---
        self._setup_vlm(
            config.vlm_path,
            config.freeze_vision_tower,
            config.freeze_florence,
            config.freeze_embeddings_only,
        )
        
        # --- Setup Tokenizer ---
        self._setup_action_tokenizer(config)

        self.rollout_step_counter = 0
        self.pred_action_seq_norm = None
        self.pred_action_seq_env = None
        self.ensure_device_consistency()

    def _init_modalities(self, config):
        self.target_modality = config.target_modality
        self.obs_modalities = config.obs_modalities
        self.img_modalities = config.img_modalities
        self.lang_modalities = config.lang_modalities

    def _init_flags(self, config):
        self.use_second_view = config.use_second_view
        self.token_dropout = config.token_dropout
        # self.use_proprio = config.use_proprio
        self.return_act_chunk = config.return_act_chunk
        self.second_view_key = config.second_view_key
        self.text_max_length = config.text_max_length
        self.prompt_robot_name = config.prompt_robot_name
        self.prompt_num_arms = config.prompt_num_arms
        self.prompt_action_space = config.prompt_action_space
        self.prompt_include_meta = config.prompt_include_meta
        self.image_resize_hw = tuple(config.image_resize_hw)
        self.image_use_clip_normalization = config.image_use_clip_normalization
        self.image_mean = tuple(CLIP_IMAGE_MEAN)
        self.image_std = tuple(CLIP_IMAGE_STD)
        self._logged_prompt_example = False

    def _setup_vlm(self, vlm_path, freeze_vision, freeze_florence, freeze_embed):
        logger.info(f"Loading VLM from {vlm_path}")

        vlm_config = AutoConfig.from_pretrained(vlm_path, trust_remote_code=True)
        vlm_config._attn_implementation = "eager"
        if getattr(vlm_config, "text_config", None) is not None:
            vlm_config.text_config._attn_implementation = "eager"

        self.vlm = AutoModelForCausalLM.from_pretrained(
            vlm_path,
            config=vlm_config,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        
        if freeze_florence:
            for param in self.vlm.parameters(): param.requires_grad = False
        elif freeze_embed:
            for param in self.vlm.get_input_embeddings().parameters(): param.requires_grad = False

        if not freeze_vision:
            for param in self.vlm.vision_tower.parameters(): param.requires_grad = True

        self.processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        
        self.prompt_embeds = self._create_prompt_embed("<Primitives>").to(self.device)
        self.vlm_vocab_size = self.vlm.language_model.get_output_embeddings().weight.shape[0] - 1

    def _setup_action_tokenizer(self, config: BeastVLAConfig) -> None:
        """
        Initializes BeastTokenizer and updates VLM vocabulary to include action tokens.
        """
        # 1. Initialize the provided BeastTokenizer
        # Note: config.action_bins should be in config (e.g., 256)
        self.action_bins = getattr(config, "action_bins", 256)
        
        self.action_tokenizer = BeastTokenizer(
            num_dof=config.num_dof,
            num_basis=config.num_basis,
            seq_len=config.act_window_size,
            vocab_size=self.action_bins,  # Beast divides range into this many bins
            degree_p=getattr(config, "degree_p", 4),
            gripper_zero_order=config.gripper_zero_order,
            gripper_dof=config.gripper_dof,
            enforce_init_pos=config.enforce_init_pos,
            device=self.device,
        )
        self.update_w_bound = config.update_w_bound
        logger.info("Using tail-of-vocabulary mapping for BEAST action tokens.")

    def _create_prompt_embed(self, prompt_text: str) -> nn.Parameter:
        self.tokenizer.add_special_tokens({"additional_special_tokens": [prompt_text]})
        self.vlm.resize_token_embeddings(len(self.tokenizer))
        prompt_token_id = self.tokenizer.convert_tokens_to_ids(prompt_text)
        # Create frozen embedding
        prompt_embed = nn.Parameter(
            self.vlm.get_input_embeddings()(torch.tensor(prompt_token_id)),
            requires_grad=False,
        )
        return prompt_embed.unsqueeze(0).unsqueeze(0)

    def ensure_device_consistency(self) -> None:
        self.to(self.device)
        self.vlm.to(self.device)
        self.action_tokenizer.to(self.device) # Ensure tokenizer buffers are on device

    def _bins_to_llm_ids(self, bin_ids: torch.Tensor) -> torch.Tensor:
        """Convert BeastTokenizer bins to VLM token IDs."""
        return self.vlm_vocab_size - 1 - bin_ids

    def _llm_ids_to_bins(self, llm_ids: torch.Tensor) -> torch.Tensor:
        """Convert VLM token IDs back to BeastTokenizer bins."""
        bins = self.vlm_vocab_size - 1 - llm_ids
        # Clamp to ensure validity during early training/sampling noise
        return torch.clamp(bins, 0, self.action_bins - 1)

    def _preprocess_images(
        self,
        image_tensor: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        image_tensor = image_tensor.to(device=device, dtype=dtype)
        image_tensor = F.interpolate(
            image_tensor,
            size=self.image_resize_hw,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        if self.image_use_clip_normalization:
            mean = torch.tensor(self.image_mean, device=device, dtype=dtype).view(1, -1, 1, 1)
            std = torch.tensor(self.image_std, device=device, dtype=dtype).view(1, -1, 1, 1)
            image_tensor = (image_tensor - mean) / std
        return image_tensor

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 1. Encode Observations
        encoded = self.encode_observations(batch)
        features = encoded['features']
        encoder_attn_mask = encoded['attention_mask']

        # 2. Prepare Targets
        actions = batch[self.target_modality]

        ### test: visualize reconstructed errors
        # self.action_tokenizer.visualize_reconstruction_error_discrete(actions)
        
        # Encode: Continuous Actions -> Discrete Bins (0..255)
        # BeastTokenizer.encode_discrete returns just the tokens tensor
        action_bins = self.action_tokenizer.encode_discrete(
            actions, 
            update_bounds=self.update_w_bound
        )
        
        # Map Bins -> VLM Input IDs
        llm_label_ids = self._bins_to_llm_ids(action_bins)

        # 3. Prepare Decoder Input
        # Use a filler token (e.g., middle of action range) as prompt for decoder
        # This matches the "empty action token" logic from original code
        B, SeqLen = llm_label_ids.shape
        filler_bin = torch.full((B, SeqLen), self.action_bins // 2, dtype=torch.long, device=self.device)
        llm_input_ids = self._bins_to_llm_ids(filler_bin)

        # 4. Bidirectional Mask
        bidirectional_mask = create_bidirectional_mask(
            batch_size=B, seq_length=SeqLen, device=self.device
        )

        # 5. Forward
        decoder_outputs = self.vlm.get_decoder()(
            input_ids=llm_input_ids,
            encoder_hidden_states=features,
            encoder_attention_mask=encoder_attn_mask,
            attention_mask=bidirectional_mask,
            use_cache=False,
        )

        lm_logits = self.vlm.language_model.get_output_embeddings()(decoder_outputs[0])
        lm_logits = lm_logits + self.vlm.language_model.final_logits_bias.to(lm_logits.device)

        # 6. Loss
        loss_fct = nn.CrossEntropyLoss()
        # View: [Batch*Seq, VocabSize] vs [Batch*Seq]
        masked_lm_loss = loss_fct(
            lm_logits.view(-1, self.vlm.config.vocab_size),
            llm_label_ids.view(-1),
        )

        # 7. Metrics (Optional)
        with torch.no_grad():
            pred_ids = torch.argmax(lm_logits, dim=-1)
            pred_bins = self._llm_ids_to_bins(pred_ids)
            # Decode: Discrete Bins -> Continuous Actions
            recon_actions = self.action_tokenizer.decode_discrete(pred_bins)
            mse = F.mse_loss(recon_actions, actions).item()
            token_pred_acc = token_prediction_accuracy(pred_ids, llm_label_ids)

        return {
            "loss": masked_lm_loss,
            "loss_dict": {"ce_loss": masked_lm_loss.item(), "mse": mse, "token_acc": token_pred_acc},
        }

    def encode_observations(self, batch: Dict) -> Dict[str, torch.Tensor]:
        device = self.device
        default_dtype = next(self.parameters()).dtype
        
        # Image encoding logic (simplified for brevity, assuming LeRobot keys)
        img_key = self.img_modalities[0] if self.img_modalities else "observation.images.image"
        if img_key not in batch and "observation.images.right_cam" in batch:
            img_key = "observation.images.right_cam"
            
        image_tensor = batch[img_key]
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor.unsqueeze(1)
        B, T, C, H, W = image_tensor.shape
        image_tensor = self._preprocess_images(
            image_tensor.reshape(-1, C, H, W),
            device=device,
            dtype=default_dtype,
        )
        
        image_features = self.vlm._encode_image(
            image_tensor
        )
        image_features = image_features.view(B, T * image_features.shape[1], -1)

        # Second view
        if self.use_second_view and self.second_view_key in batch:
            img2 = batch[self.second_view_key]
            if len(img2.shape) == 4: img2 = img2.unsqueeze(1)
            feat2 = self.vlm._encode_image(
                self._preprocess_images(
                    img2.reshape(-1, C, H, W),
                    device=device,
                    dtype=default_dtype,
                )
            )
            feat2 = feat2.view(B, T * feat2.shape[1], -1)
            image_features = torch.cat([image_features, feat2], dim=1)

        # Text encoding
        txt = batch.get("task", self.task)
        if isinstance(txt, tuple):
            txt = list(txt)
        elif isinstance(txt, str):
            txt = [txt] * B
        elif not isinstance(txt, list):
            txt = [self.task] * B

        prompts = [
            build_policy_prompt(
                instruction=instruction,
                robot_name=self.prompt_robot_name,
                num_arms=self.prompt_num_arms,
                action_space=self.prompt_action_space,
                include_meta=self.prompt_include_meta,
            )
            for instruction in txt
        ]
        if not self._logged_prompt_example and prompts:
            logger.info("BEAST task example | raw: %s | prompt: %s", txt[0], prompts[0])
            self._logged_prompt_example = True
        
        tokens = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.text_max_length,
        ).to(device)
        
        text_embeds = self.vlm.get_input_embeddings()(tokens["input_ids"])
        
        # Combine
        task_prompt = self.prompt_embeds.expand(B, -1, -1)
        merged = torch.cat([image_features, task_prompt, text_embeds], dim=1)
        attn_mask = torch.ones(merged.shape[:2], dtype=torch.long, device=device)

        features = self.vlm.get_encoder()(
            inputs_embeds=merged, attention_mask=attn_mask
        ).last_hidden_state
        
        return {"features": features, "attention_mask": attn_mask}

    def sample_actions(self, z, cond, inference=False):
        features = cond["features"]
        mask = cond["attention_mask"]
        B = features.shape[0]
        
        # 1. Construct Filler Input
        # We need (NumDOF * NumBasis) tokens
        seq_len_tokens = self.action_tokenizer.num_dof * self.action_tokenizer.num_basis
        filler_bin = torch.full((B, seq_len_tokens), self.action_bins // 2, dtype=torch.long, device=self.device)
        llm_input_ids = self._bins_to_llm_ids(filler_bin)
        
        # 2. Bidirectional Mask
        bidirectional_mask = create_bidirectional_mask(B, seq_len_tokens, self.device)
        
        # 3. Decode
        decoder_outputs = self.vlm.get_decoder()(
            input_ids=llm_input_ids,
            encoder_hidden_states=features,
            encoder_attention_mask=mask,
            attention_mask=bidirectional_mask,
            use_cache=False,
        )
        
        lm_logits = self.vlm.language_model.get_output_embeddings()(decoder_outputs[0])
        lm_logits = lm_logits + self.vlm.language_model.final_logits_bias.to(lm_logits.device)
        
        # 4. Reconstruct
        pred_ids = torch.argmax(lm_logits, dim=-1)
        pred_bins = self._llm_ids_to_bins(pred_ids)
        
        #(optional) print continuous control point 
        control_points_flat = discrete_to_continuous(
            einops.rearrange(
                pred_bins,
                "b (t d) -> b (d t)",
                t=self.action_tokenizer.num_basis,
                d=self.action_tokenizer.num_dof,
            ),
            min_val=self.action_tokenizer.w_min,
            max_val=self.action_tokenizer.w_max,
            num_bins=self.action_bins,
        )

        control_points = einops.rearrange(
            control_points_flat,
            "b (d t) -> b t d",
            d=self.action_tokenizer.num_dof,
            t=self.action_tokenizer.num_basis,
        )

        print("=== New BEAST chunk ===")
        print("control_points[0] shape:", tuple(control_points[0].shape))
        print(control_points[0].detach().cpu())
        
        # Use init_pos relative reconstruction if needed (logic from original beast_florence)
        # beast.py decode_discrete accepts init_pos
        init_pos = None
        if self.pred_action_seq_norm is not None and self.action_tokenizer.enforce_init_pos:
            # Use last action of previous chunk as start of next
            init_pos = self.pred_action_seq_norm[:, -1, ...]
            
        actions = self.action_tokenizer.decode_discrete(pred_bins, init_pos=init_pos)
        
        return actions

    def reset(self):
        self.rollout_step_counter = 0
        self.pred_action_seq_norm = None
        self.pred_action_seq_env = None
        self.eval()
