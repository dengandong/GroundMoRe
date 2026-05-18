"""
MoRA with a Qwen3-VL/Qwen-VL backbone.

This file intentionally keeps the Qwen path separate from model/MoRA.py so the
existing LLaVA/LISA experiments stay unchanged.  Qwen3-VL requires a recent
Transformers release; the current cluster env may only expose Qwen2-VL, so the
base class is resolved lazily with Qwen3-VL preferred and Qwen2-VL as an API
compatible fallback for syntax/import checks.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from .segment_anything import build_sam_vit_h


QwenVLForConditionalGeneration = getattr(
    transformers, "Qwen3VLForConditionalGeneration", None
) or getattr(transformers, "Qwen2VLForConditionalGeneration", None)

if QwenVLForConditionalGeneration is None:
    raise ImportError(
        "MoRA_Qwen3 needs transformers with Qwen3VLForConditionalGeneration "
        "(preferred) or Qwen2VLForConditionalGeneration. Please upgrade "
        "transformers before importing this model."
    )


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,
    eps=1e-6,
):
    """Compute DICE loss for mask supervision."""
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    return loss.sum() / (num_masks + 1e-8)


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)


def _config_hidden_size(config) -> int:
    if hasattr(config, "hidden_size"):
        return config.hidden_size
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return config.text_config.hidden_size
    raise AttributeError("Cannot find hidden_size on Qwen-VL config/text_config.")


class MoRAQwen3ForConditionalGeneration(QwenVLForConditionalGeneration):
    """Qwen-VL backbone plus MoRA's SAM decoder and temporal localization heads."""

    def __init__(self, config, **kwargs):
        self.ce_loss_weight = kwargs.pop(
            "ce_loss_weight", getattr(config, "ce_loss_weight", 1.0)
        )
        self.dice_loss_weight = kwargs.pop(
            "dice_loss_weight", getattr(config, "dice_loss_weight", 0.5)
        )
        self.bce_loss_weight = kwargs.pop(
            "bce_loss_weight", getattr(config, "bce_loss_weight", 2.0)
        )
        self.loc_loss_weight = kwargs.pop(
            "loc_loss_weight", getattr(config, "loc_loss_weight", 1.0)
        )
        self.global_loc_loss_weight = kwargs.pop(
            "global_loc_loss_weight", getattr(config, "global_loc_loss_weight", 1.0)
        )
        self.loc_mode = kwargs.pop("loc_mode", getattr(config, "loc_mode", "global"))
        self.seg_token_idx = kwargs.pop(
            "seg_token_idx", getattr(config, "seg_token_idx", None)
        )
        self.loc_token_idx = kwargs.pop(
            "loc_token_idx", getattr(config, "loc_token_idx", None)
        )
        self.use_loc_token = kwargs.pop(
            "use_loc_token", getattr(config, "use_loc_token", False)
        )
        self.train_mask_decoder = kwargs.pop(
            "train_mask_decoder", getattr(config, "train_mask_decoder", True)
        )
        self.out_dim = kwargs.pop("out_dim", getattr(config, "out_dim", 256))
        self.vision_pretrained = kwargs.pop(
            "vision_pretrained", getattr(config, "vision_pretrained", None)
        )

        config.ce_loss_weight = self.ce_loss_weight
        config.dice_loss_weight = self.dice_loss_weight
        config.bce_loss_weight = self.bce_loss_weight
        config.loc_loss_weight = self.loc_loss_weight
        config.global_loc_loss_weight = self.global_loc_loss_weight
        config.loc_mode = self.loc_mode
        config.seg_token_idx = self.seg_token_idx
        config.loc_token_idx = self.loc_token_idx
        config.use_loc_token = self.use_loc_token
        config.train_mask_decoder = self.train_mask_decoder
        config.out_dim = self.out_dim
        config.vision_pretrained = self.vision_pretrained
        config.use_cache = False

        super().__init__(config)
        self.initialize_mora_modules()

    def initialize_mora_modules(self):
        self.initialize_visual_model()

        hidden_size = _config_hidden_size(self.config)
        text_fc = [
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, self.out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

        if self.use_loc_token:
            self.loc_temporal_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, 1),
            )
            self.loc_global_context_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
            )
            self.loc_global_temporal_head = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, 1),
            )
            for module in (
                self.loc_temporal_head,
                self.loc_global_context_proj,
                self.loc_global_temporal_head,
            ):
                module.train()
                for param in module.parameters():
                    param.requires_grad = True

    def initialize_visual_model(self):
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if self.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

    def reload_visual_model(self, vision_pretrained=None):
        if vision_pretrained is not None:
            self.vision_pretrained = vision_pretrained
            self.config.vision_pretrained = vision_pretrained
        self.initialize_visual_model()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.view(
                -1, pixel_values.size(-3), pixel_values.size(-2), pixel_values.size(-1)
            )
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def _repeat_qwen_image_inputs(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.LongTensor,
        offset: torch.LongTensor,
        batch_size: int,
        t_size: int,
        target_rows: int,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Repeat Qwen per-frame image patches for each expression in a video."""
        if pixel_values is None or image_grid_thw is None:
            return pixel_values, image_grid_thw
        if image_grid_thw.size(0) == target_rows:
            return pixel_values, image_grid_thw
        if image_grid_thw.size(0) != batch_size * t_size:
            raise ValueError(
                "image_grid_thw must describe either b*t original frames or "
                f"b*K*t model rows, got {image_grid_thw.size(0)} grids for "
                f"batch={batch_size}, frames={t_size}, target_rows={target_rows}."
            )

        patch_counts = image_grid_thw.prod(dim=1).tolist()
        pixel_chunks = list(torch.split(pixel_values, patch_counts, dim=0))
        repeated_pixels = []
        repeated_grids = []
        for batch_idx in range(batch_size):
            num_expr = int((offset[batch_idx + 1] - offset[batch_idx]).item())
            frame_start = batch_idx * t_size
            for _ in range(num_expr):
                for frame_idx in range(frame_start, frame_start + t_size):
                    repeated_pixels.append(pixel_chunks[frame_idx])
                    repeated_grids.append(image_grid_thw[frame_idx])
        return torch.cat(repeated_pixels, dim=0), torch.stack(repeated_grids, dim=0)

    @staticmethod
    def _shifted_token_mask(input_ids: torch.LongTensor, token_idx: int) -> torch.Tensor:
        token_mask = input_ids[:, 1:] == token_idx
        return torch.cat(
            [
                token_mask,
                torch.zeros(
                    (token_mask.shape[0], 1),
                    dtype=torch.bool,
                    device=token_mask.device,
                ),
            ],
            dim=1,
        )

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        loc_targets_list: Optional[List[torch.Tensor]] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        inference: bool = False,
        **kwargs,
    ):
        if pixel_values_videos is not None or video_grid_thw is not None:
            raise NotImplementedError(
                "MoRA-Qwen3 currently expects each sampled video frame to be "
                "processed as an image via pixel_values/image_grid_thw. The "
                "video_grid_thw path can be added later if we train with Qwen's "
                "native video processor."
            )
        if pixel_values is None or image_grid_thw is None:
            raise ValueError(
                "MoRA-Qwen3 needs Qwen processor outputs: pixel_values and "
                "image_grid_thw. The old LLaVA images_clip tensor is not used."
            )

        questions_list = kwargs.get("questions_list")
        batch_size = images.shape[0]
        t_size = images.size(1)
        bK_size = int(offset[-1].item())
        assert batch_size == len(offset) - 1

        input_ids = input_ids.unsqueeze(1).repeat(1, t_size, 1)
        input_ids = input_ids.view(-1, input_ids.size(2))
        attention_masks = attention_masks.unsqueeze(1).repeat(1, t_size, 1)
        attention_masks = attention_masks.view(-1, attention_masks.size(2))
        labels = labels.unsqueeze(1).repeat(1, t_size, 1)
        labels = labels.view(-1, labels.size(2))
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.unsqueeze(1).repeat(1, t_size, 1)
            mm_token_type_ids = mm_token_type_ids.view(-1, mm_token_type_ids.size(2))
        label_list = label_list * t_size
        resize_list = resize_list * t_size

        pixel_values, image_grid_thw = self._repeat_qwen_image_inputs(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            offset=offset,
            batch_size=batch_size,
            t_size=t_size,
            target_rows=input_ids.size(0),
        )

        frames_bt = images.view(-1, images.size(2), images.size(3), images.size(4))
        image_embeddings = self.get_visual_embs(frames_bt)

        seg_token_mask = self._shifted_token_mask(input_ids, self.seg_token_idx)
        loc_token_mask = None
        if self.use_loc_token and self.loc_token_idx is not None:
            loc_token_mask = self._shifted_token_mask(input_ids, self.loc_token_idx)

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_masks,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        output_hidden_states = output.hidden_states
        token_hidden_states = output_hidden_states[-1]

        assert len(self.text_hidden_fcs) == 1
        projected_hidden_states = self.text_hidden_fcs[0](token_hidden_states)
        pred_embeddings = projected_hidden_states[seg_token_mask]
        if pred_embeddings.size(0) < bK_size * t_size:
            raise RuntimeError(
                f"Expected at least {bK_size * t_size} [SEG] embeddings, "
                f"found {pred_embeddings.size(0)}. Check Qwen prompt/tokenization."
            )
        pred_embeddings = pred_embeddings[: bK_size * t_size].view(bK_size, t_size, -1)

        pred_embeddings_by_video = []
        cursor = 0
        for batch_idx in range(batch_size):
            num_expr = int((offset[batch_idx + 1] - offset[batch_idx]).item())
            pred_embeddings_i = pred_embeddings[cursor : cursor + num_expr]
            pred_embeddings_by_video.append(pred_embeddings_i.reshape(num_expr * t_size, -1))
            cursor += num_expr

        pred_masks = []
        for batch_idx, pred_embedding in enumerate(pred_embeddings_by_video):
            num_expr = max(1, pred_embedding.size(0) // t_size)
            image_start = batch_idx * t_size
            image_end = image_start + t_size
            image_embeddings_i = image_embeddings[image_start:image_end]
            if num_expr > 1:
                image_embeddings_i = image_embeddings_i.repeat(num_expr, 1, 1, 1)

            sparse_embeddings, dense_embeddings = self.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embedding.unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embedding.dtype)
            low_res_masks, _ = self.visual_model.mask_decoder(
                image_embeddings=image_embeddings_i,
                image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            pred_mask = self.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[batch_idx],
                original_size=label_list[batch_idx].shape,
            )
            pred_masks.append(pred_mask[:, 0])

        gt_masks = masks_list
        loc_probs = None
        if self.use_loc_token and loc_token_mask is not None:
            loc_hidden_states = token_hidden_states[loc_token_mask]
            expected_loc_tokens = bK_size * t_size
            if loc_hidden_states.shape[0] >= expected_loc_tokens:
                loc_hidden_states = loc_hidden_states[:expected_loc_tokens].view(
                    bK_size, t_size, -1
                )
                if self.loc_mode == "frame":
                    loc_logits_for_inference = self.loc_temporal_head(
                        loc_hidden_states
                    ).squeeze(-1)
                    loc_probs = torch.sigmoid(loc_logits_for_inference)
                elif self.loc_mode == "global":
                    global_context = self.loc_global_context_proj(
                        loc_hidden_states.mean(dim=1, keepdim=True)
                    ).expand(-1, t_size, -1)
                    global_loc_input = torch.cat(
                        [loc_hidden_states, global_context], dim=-1
                    )
                    loc_logits_for_inference = self.loc_global_temporal_head(
                        global_loc_input
                    ).squeeze(-1)
                    loc_probs = torch.sigmoid(loc_logits_for_inference)
        if inference:
            output_dict = {"pred_masks": pred_masks, "gt_masks": gt_masks}
            if loc_probs is not None:
                output_dict["loc_probs"] = loc_probs
            return output_dict

        ce_loss = output.loss * self.ce_loss_weight
        loss = ce_loss + 0
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]
            assert gt_mask.shape[0] == pred_mask.shape[0], (
                f"gt_mask.shape: {gt_mask.shape}, pred_mask.shape: {pred_mask.shape}"
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss
        loss += mask_loss

        loc_loss = torch.zeros_like(loss)
        loc_frame_loss = torch.zeros_like(loss)
        loc_global_loss = torch.zeros_like(loss)
        if (
            self.use_loc_token
            and loc_token_mask is not None
            and loc_targets_list is not None
        ):
            loc_hidden_states = token_hidden_states[loc_token_mask]
            expected_loc_tokens = bK_size * t_size
            if loc_hidden_states.shape[0] >= expected_loc_tokens:
                loc_hidden_states = loc_hidden_states[:expected_loc_tokens].view(
                    bK_size, t_size, -1
                )
                loc_targets = []
                for batch_idx in range(batch_size):
                    num_expr = int((offset[batch_idx + 1] - offset[batch_idx]).item())
                    loc_target = loc_targets_list[batch_idx].to(
                        device=loc_hidden_states.device,
                        dtype=loc_hidden_states.dtype,
                    )
                    if loc_target.numel() < t_size:
                        loc_target = F.pad(loc_target, (0, t_size - loc_target.numel()))
                    elif loc_target.numel() > t_size:
                        loc_target = loc_target[:t_size]
                    loc_targets.append(loc_target.unsqueeze(0).expand(num_expr, -1))
                loc_targets = torch.cat(loc_targets, dim=0)

                if self.loc_mode == "frame" and self.loc_loss_weight > 0:
                    loc_logits = self.loc_temporal_head(loc_hidden_states).squeeze(-1)
                    loc_frame_loss = F.binary_cross_entropy_with_logits(
                        loc_logits, loc_targets, reduction="mean"
                    )
                    loc_frame_loss = loc_frame_loss * self.loc_loss_weight

                if self.loc_mode == "global" and self.global_loc_loss_weight > 0:
                    global_context = self.loc_global_context_proj(
                        loc_hidden_states.mean(dim=1, keepdim=True)
                    ).expand(-1, t_size, -1)
                    global_loc_input = torch.cat(
                        [loc_hidden_states, global_context], dim=-1
                    )
                    global_loc_logits = self.loc_global_temporal_head(
                        global_loc_input
                    ).squeeze(-1)
                    loc_global_loss = F.binary_cross_entropy_with_logits(
                        global_loc_logits, loc_targets, reduction="mean"
                    )
                    loc_global_loss = loc_global_loss * self.global_loc_loss_weight

                loc_loss = loc_frame_loss + loc_global_loss
                loss += loc_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
            "loc_loss": loc_loss,
            "loc_frame_loss": loc_frame_loss,
            "loc_global_loss": loc_global_loss,
            "raw_frames": images[0],
            "pred_masks": pred_masks[0],
            "gt_masks": gt_masks[0],
            "questions": questions_list[0][0] if questions_list else "",
        }


# Short aliases for future training scripts/checkpoints.
MoRAQwen3ForCausalLM = MoRAQwen3ForConditionalGeneration
MoRAQwenVLForConditionalGeneration = MoRAQwen3ForConditionalGeneration
