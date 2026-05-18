"""
This is modified from LISA.py that changing the original single-frame input as multi
"""
from typing import List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM, LlavaLlamaModel)
# from .llava.model.language_model.llava_mpt import (LlavaMPTForCausalLM, LlavaMPTModel)

from .segment_anything import build_sam_vit_h
import cv2
import numpy as np


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class MoRAMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(MoRAMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

        if getattr(config, "use_loc_token", False):
            self.loc_temporal_head = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, 1),
            )
            self.loc_temporal_head.train()
            for param in self.loc_temporal_head.parameters():
                param.requires_grad = True
            self.loc_global_context_proj = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
            )
            self.loc_global_temporal_head = nn.Sequential(
                nn.Linear(in_dim * 2, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, 1),
            )
            self.loc_global_context_proj.train()
            self.loc_global_temporal_head.train()
            for param in self.loc_global_context_proj.parameters():
                param.requires_grad = True
            for param in self.loc_global_temporal_head.parameters():
                param.requires_grad = True


class MoRAModel(MoRAMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(MoRAModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class MoRAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )

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
        config.ce_loss_weight = self.ce_loss_weight
        config.dice_loss_weight = self.dice_loss_weight
        config.bce_loss_weight = self.bce_loss_weight
        config.loc_loss_weight = self.loc_loss_weight
        config.global_loc_loss_weight = self.global_loc_loss_weight
        config.loc_mode = self.loc_mode
        config.seg_token_idx = self.seg_token_idx
        config.loc_token_idx = self.loc_token_idx
        config.use_loc_token = self.use_loc_token

        super().__init__(config)

        self.model = MoRAModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.view(
                -1, pixel_values.size(-3), pixel_values.size(-2), pixel_values.size(-1)
            )
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,  # size=[b, t, c, h, w]
        images_clip: torch.FloatTensor,  # processed input for CLIP image features, [b, t, 3, 224, 224]
        input_ids: torch.LongTensor,  # tokenized ids, need to be duplicated for t times
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,  # index for seg token? related to number of expression for each sample
        masks_list: List[torch.FloatTensor],  # len==b --> len == b, [t, xx]
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        loc_targets_list: List[torch.Tensor] = None,
        inference: bool = False,
        **kwargs,
    ):
        
        questions_list = kwargs.get('questions_list')

        # repeat n_frames time
        batch_size = images.shape[0]
        t_size = images.size(1)
        assert batch_size == len(offset) - 1
        
        # duplicate t times for the language part
        input_ids = input_ids.unsqueeze(1).repeat(1, t_size, 1)
        input_ids = input_ids.view(-1, input_ids.size(2))  # Kt, 83
        attention_masks = attention_masks.unsqueeze(1).repeat(1, t_size, 1)
        attention_masks = attention_masks.view(-1, attention_masks.size(2))  # Kt, 83
        labels = labels.unsqueeze(1).repeat(1, t_size, 1)
        labels = labels.view(-1, labels.size(2))  # Kt, 83
        label_list = label_list * t_size
        resize_list = resize_list * t_size

        frames_bt = images.view(-1, images.size(2), images.size(3), images.size(4))  # bt, c, h, w
        image_embeddings = self.get_visual_embs(frames_bt)  # bt, dim

        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        # seg_token_mask_new = input_ids[:, :] == self.seg_token_idx
        # img_token_mask = input_ids[:, :] == -200 # IMAGE_TOKEN_INDEX

        # mix_token_mask = seg_token_mask_new | img_token_mask
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
            dim=1,
        )
        loc_token_mask = None
        if self.use_loc_token and self.loc_token_idx is not None:
            loc_token_mask = input_ids[:, 1:] == self.loc_token_idx
            loc_token_mask = torch.cat(
                [
                    loc_token_mask,
                    torch.zeros((loc_token_mask.shape[0], 1)).bool().cuda(),
                ],
                dim=1,
            )
            loc_token_mask = torch.cat(
                [
                    torch.zeros((loc_token_mask.shape[0], 255)).bool().cuda(),
                    loc_token_mask,
                ],
                dim=1,
            )

        ###
        # input_ids = input_ids * mix_token_mask  #  QA+[SEG]+[IMAGE]

        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None

        else:
            images_clip_list = []
            for i in range(len(offset) - 1):  # for loop w.r.t. batch size
                start_i, end_i = offset[i], offset[i + 1]  # end - start = # sentence for one case
                images_clip_i = (
                    images_clip[i]  # t, 3, 224, 224
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1, -1)  # K=1, t, 3, 224, 224
                    .contiguous()
                )  
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)  # b1_K+b2_K:bK, t, 3, 224, 224
            bK_size = images_clip.size(0)
            images_clip = images_clip.view(-1, images_clip.size(2), images_clip.size(3), images_clip.size(4))  # bKt, 3, 224, 224

            output = super().forward(
                images=images_clip, # [K, 3, 224, 224]
                attention_mask=attention_masks,
                input_ids=input_ids, # [K, 83] K, dd
                labels=labels,
                output_hidden_states=True,
            ) 

            # output_raw = super().forward(
            #     images=images_clip, # [K, 3, 224, 224]
            #     attention_mask=attention_masks,
            #     input_ids=input_ids, # [K, 83] K, dd
            #     labels=labels,
            #     output_hidden_states=True,
            # )
            output_hidden_states = output.hidden_states  # is a tuple 

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))  # bkt, L, 256
        # hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))  # evaluation !!!

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]  # bKt  size: num_frame,  4dd, 256; num_frame, 4dd
        pred_embeddings = pred_embeddings.view(bK_size, t_size, -1)  # bK, t, dim
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bt, ]

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )

        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_i = pred_embeddings[start_i:end_i]  # [K, t, dim]
            K_size = pred_embeddings_i.size(0)
            pred_embeddings_.append(pred_embeddings_i.view(K_size*t_size, -1))
        pred_embeddings = pred_embeddings_   # list, each of the element is for one video, each element is [K*t, dim]

        multimask_output = False
        pred_masks = []
        for i in range(len(pred_embeddings)):
            K_size = max(1, pred_embeddings[i].size(0) // t_size)
            image_start = i * t_size
            image_end = image_start + t_size
            image_embeddings_i = image_embeddings[image_start:image_end]
            if image_embeddings_i.size(0) != t_size:
                image_embeddings_i = image_embeddings[:t_size]
            if K_size > 1:
                image_embeddings_i = image_embeddings_i.repeat(K_size, 1, 1, 1)
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings_i,
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])

        model_output = output
        gt_masks = masks_list

        # # ========================== temporarily for visualization ==================== #
        # print("saving mask predictions ... ")
        # random_id = int(torch.randint(low=0, high=10000, size=(1,))[0])

        # for i in range(pred_masks[0].size(0)):
        #     pred_m = pred_masks[0][i].detach().cpu().numpy()
        #     gt_m = gt_masks[0][i].detach().cpu().numpy()

        #     pred_normalized = cv2.normalize(pred_m, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #     _, binary_pred = cv2.threshold(pred_normalized, 127, 255, cv2.THRESH_BINARY)

        #     gt_normalized = cv2.normalize(gt_m, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #     _, binary_gt = cv2.threshold(gt_normalized, 127, 255, cv2.THRESH_BINARY)

        #     cv2.imwrite("examples/pred_mask_{}_{}.png".format(random_id, i), binary_pred)
        #     cv2.imwrite("examples/gt_mask_{}_{}.png".format(random_id, i), binary_gt)
            

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        output = model_output.logits

        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        loss = ce_loss + 0
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
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
            and (
                hasattr(self.model, "loc_temporal_head")
                or hasattr(self.model, "loc_global_temporal_head")
            )
        ):
            loc_last_hidden_state = output_hidden_states[-1]
            if loc_token_mask.size(1) < loc_last_hidden_state.size(1):
                pad = torch.zeros(
                    (
                        loc_token_mask.size(0),
                        loc_last_hidden_state.size(1) - loc_token_mask.size(1),
                    ),
                    dtype=torch.bool,
                    device=loc_token_mask.device,
                )
                loc_token_mask = torch.cat([loc_token_mask, pad], dim=1)
            elif loc_token_mask.size(1) > loc_last_hidden_state.size(1):
                loc_token_mask = loc_token_mask[:, : loc_last_hidden_state.size(1)]

            loc_hidden_states = loc_last_hidden_state[loc_token_mask]
            expected_loc_tokens = bK_size * t_size
            if loc_hidden_states.shape[0] >= expected_loc_tokens:
                loc_hidden_states = loc_hidden_states[:expected_loc_tokens].view(
                    bK_size, t_size, -1
                )
                loc_targets = []
                for batch_idx in range(len(offset) - 1):
                    num_expr = int((offset[batch_idx + 1] - offset[batch_idx]).item())
                    target_device = loc_hidden_states.device
                    target_dtype = loc_hidden_states.dtype
                    loc_target = loc_targets_list[batch_idx].to(
                        device=target_device, dtype=target_dtype
                    )
                    if loc_target.numel() < t_size:
                        loc_target = F.pad(loc_target, (0, t_size - loc_target.numel()))
                    elif loc_target.numel() > t_size:
                        loc_target = loc_target[:t_size]
                    loc_targets.append(loc_target.unsqueeze(0).expand(num_expr, -1))
                loc_targets = torch.cat(loc_targets, dim=0)
                if (
                    self.loc_mode == "frame"
                    and hasattr(self.model, "loc_temporal_head")
                    and self.loc_loss_weight > 0
                ):
                    loc_logits = self.model.loc_temporal_head(loc_hidden_states).squeeze(-1)
                    loc_frame_loss = F.binary_cross_entropy_with_logits(
                        loc_logits, loc_targets, reduction="mean"
                    )
                    loc_frame_loss = loc_frame_loss * self.loc_loss_weight

                if (
                    self.loc_mode == "global"
                    and hasattr(self.model, "loc_global_context_proj")
                    and hasattr(self.model, "loc_global_temporal_head")
                    and self.global_loc_loss_weight > 0
                ):
                    global_context = self.model.loc_global_context_proj(
                        loc_hidden_states.mean(dim=1, keepdim=True)
                    ).expand(-1, t_size, -1)
                    global_loc_input = torch.cat(
                        [loc_hidden_states, global_context], dim=-1
                    )
                    global_loc_logits = self.model.loc_global_temporal_head(
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
            "questions": questions_list[0][0]
        }

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
        force_seg_token=False,
        force_loc_token=False,
        return_loc=False,
    ):
        with torch.no_grad():
            if images.dim() == 5:
                batch_size, t_size = images.shape[:2]
                frame_images = images.reshape(
                    -1, images.size(-3), images.size(-2), images.size(-1)
                )
            else:
                batch_size, t_size = input_ids.size(0), images.size(0)
                frame_images = images

            if images_clip.dim() == 5:
                gen_images_clip = images_clip[:, 0]
                frame_images_clip = images_clip.reshape(
                    -1,
                    images_clip.size(-3),
                    images_clip.size(-2),
                    images_clip.size(-1),
                )
            else:
                gen_images_clip = images_clip
                frame_images_clip = images_clip

            outputs = self.generate(
                images=gen_images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_ids = outputs.sequences

            if force_seg_token:
                has_seg = (output_ids == self.seg_token_idx).any(dim=1)
                if not has_seg.all():
                    seg_ids = torch.full(
                        (output_ids.shape[0], 1),
                        self.seg_token_idx,
                        dtype=output_ids.dtype,
                        device=output_ids.device,
                    )
                    output_ids = torch.cat([output_ids, seg_ids], dim=1)
            if (
                force_loc_token
                and self.use_loc_token
                and self.loc_token_idx is not None
            ):
                has_loc = (output_ids == self.loc_token_idx).any(dim=1)
                if not has_loc.all():
                    loc_ids = torch.full(
                        (output_ids.shape[0], 1),
                        self.loc_token_idx,
                        dtype=output_ids.dtype,
                        device=output_ids.device,
                    )
                    output_ids = torch.cat([output_ids, loc_ids], dim=1)

            frame_input_ids = output_ids.unsqueeze(1).repeat(1, t_size, 1)
            frame_input_ids = frame_input_ids.reshape(-1, output_ids.size(-1))
            frame_attention_masks = torch.ones_like(frame_input_ids, dtype=torch.bool)
            frame_outputs = super().forward(
                images=frame_images_clip,
                attention_mask=frame_attention_masks,
                input_ids=frame_input_ids,
                output_hidden_states=True,
            )
            output_hidden_states = frame_outputs.hidden_states
            if isinstance(output_hidden_states, (tuple, list)):
                output_hidden_states = output_hidden_states[-1]

            loc_probs = None
            if return_loc and self.use_loc_token and self.loc_token_idx is not None:
                loc_token_mask = frame_input_ids[:, 1:] == self.loc_token_idx
                loc_token_mask = torch.cat(
                    [
                        loc_token_mask,
                        torch.zeros(
                            (loc_token_mask.shape[0], 1),
                            dtype=torch.bool,
                            device=loc_token_mask.device,
                        ),
                    ],
                    dim=1,
                )
                loc_token_mask = torch.cat(
                    [
                        torch.zeros(
                            (loc_token_mask.shape[0], 255),
                            dtype=torch.bool,
                            device=loc_token_mask.device,
                        ),
                        loc_token_mask,
                    ],
                    dim=1,
                )
                if loc_token_mask.size(1) < output_hidden_states.size(1):
                    pad = torch.zeros(
                        (
                            loc_token_mask.size(0),
                            output_hidden_states.size(1) - loc_token_mask.size(1),
                        ),
                        dtype=torch.bool,
                        device=loc_token_mask.device,
                    )
                    loc_token_mask = torch.cat([loc_token_mask, pad], dim=1)
                elif loc_token_mask.size(1) > output_hidden_states.size(1):
                    loc_token_mask = loc_token_mask[:, : output_hidden_states.size(1)]

                loc_hidden_states = output_hidden_states[loc_token_mask]
                expected_loc_tokens = batch_size * t_size
                if loc_hidden_states.size(0) >= expected_loc_tokens:
                    loc_hidden_states = loc_hidden_states[:expected_loc_tokens].view(
                        batch_size, t_size, -1
                    )
                    if (
                        self.loc_mode == "frame"
                        and hasattr(self.model, "loc_temporal_head")
                    ):
                        loc_logits = self.model.loc_temporal_head(
                            loc_hidden_states
                        ).squeeze(-1)
                        loc_probs = loc_logits.sigmoid()
                    elif (
                        self.loc_mode == "global"
                        and hasattr(self.model, "loc_global_context_proj")
                        and hasattr(self.model, "loc_global_temporal_head")
                    ):
                        global_context = self.model.loc_global_context_proj(
                            loc_hidden_states.mean(dim=1, keepdim=True)
                        ).expand(-1, t_size, -1)
                        global_loc_input = torch.cat(
                            [loc_hidden_states, global_context], dim=-1
                        )
                        loc_logits = self.model.loc_global_temporal_head(
                            global_loc_input
                        ).squeeze(-1)
                        loc_probs = loc_logits.sigmoid()

            seg_token_mask = frame_input_ids[:, 1:] == self.seg_token_idx
            seg_token_mask = torch.cat(
                [
                    seg_token_mask,
                    torch.zeros(
                        (seg_token_mask.shape[0], 1),
                        dtype=torch.bool,
                        device=seg_token_mask.device,
                    ),
                ],
                dim=1,
            )
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            seg_token_mask = torch.cat(
                [
                    torch.zeros(
                        (seg_token_mask.shape[0], 255),
                        dtype=torch.bool,
                        device=seg_token_mask.device,
                    ),
                    seg_token_mask,
                ],
                dim=1,
            )

            assert len(self.model.text_hidden_fcs) == 1
            last_hidden_state = self.model.text_hidden_fcs[0](output_hidden_states)
            if seg_token_mask.size(1) < last_hidden_state.size(1):
                pad = torch.zeros(
                    (
                        seg_token_mask.size(0),
                        last_hidden_state.size(1) - seg_token_mask.size(1),
                    ),
                    dtype=torch.bool,
                    device=seg_token_mask.device,
                )
                seg_token_mask = torch.cat([seg_token_mask, pad], dim=1)
            elif seg_token_mask.size(1) > last_hidden_state.size(1):
                seg_token_mask = seg_token_mask[:, : last_hidden_state.size(1)]

            pred_embeddings = last_hidden_state[seg_token_mask]
            expected_prompts = batch_size * t_size
            if pred_embeddings.size(0) < expected_prompts:
                if return_loc:
                    return output_ids, [], loc_probs
                return output_ids, []
            pred_embeddings = pred_embeddings[:expected_prompts].reshape(
                batch_size, t_size, -1
            )

            image_embeddings = self.get_visual_embs(frame_images)

            multimask_output = False
            pred_masks = []
            for i in range(batch_size):
                for t in range(t_size):
                    frame_idx = i * t_size + t
                    frame_pred_embedding = pred_embeddings[i, t].view(1, 1, -1)
                    (
                        sparse_embeddings,
                        dense_embeddings,
                    ) = self.model.visual_model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=frame_pred_embedding,
                    )

                    sparse_embeddings = sparse_embeddings.to(frame_pred_embedding.dtype)
                    low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                        image_embeddings=image_embeddings[frame_idx].unsqueeze(0),
                        image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=multimask_output,
                    )
                    pred_mask = self.model.visual_model.postprocess_masks(
                        low_res_masks,
                        input_size=resize_list[min(frame_idx, len(resize_list) - 1)],
                        original_size=original_size_list[
                            min(frame_idx, len(original_size_list) - 1)
                        ],
                    )
                    pred_masks.append(pred_mask[:, 0])

        if return_loc:
            return output_ids, pred_masks, loc_probs
        return output_ids, pred_masks


# Backward-compatible names used by older scripts/checkpoints.
VideoLISAForCausalLM = MoRAForCausalLM
PooledVideoLISAForCausalLM = MoRAForCausalLM
