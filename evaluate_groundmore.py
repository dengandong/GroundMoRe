import argparse
import os
import sys
import json
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.MoRA import VideoLISAForCausalLM, PooledVideoLISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, LONG_QUESTION_LIST,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, REASONING_LONG_QUESTION_LIST)

from davis2017.metrics import db_eval_boundary, db_eval_iou
from davis2017.utils import db_statistics


def time_str_to_seconds(time_str):
    """Converts a time string to seconds."""
    parts = time_str.split(":")
    parts = [int(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]



def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA chat")
    parser.add_argument("--version", default="MoRA-ZA-7B")
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def main(args):
    args = parse_args(args)

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]


    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = PooledVideoLISAForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()

    # initialize the J&F res
    J_list = []
    F_list = []

    causal_J_list = []
    causal_F_list = []
    sequential_J_list = []
    sequential_F_list = []
    counterfactual_J_list = []
    counterfactual_F_list = []
    descriptive_J_list = []
    descriptive_F_list = []

    # load data
    video_root = "./groundmore_videos/"
    meta_file = "groundmore_test.json"
    save_path_prefix = "output/"

    file = open('mora-zs-7b.txt', 'w')

    with open(meta_file, "r") as f:
        metadata = json.load(f)["videos"]

    video_list = list(metadata.keys())

    # 1. For each video
    for video in tqdm(video_list):
        metas = [] # list[dict], length is number of expressions

        expressions = metadata[video]["questions"]   
        expression_list = list(expressions.keys()) 
        num_expressions = len(expression_list)

        clip_start = video[-9:].split("_")[0][:2] + ":" + video[-9:].split("_")[0][2:]
        clip_end = video[-9:].split("_")[1][:2] + ":" + video[-9:].split("_")[1][2:]

        # read all the anno meta
        for i in range(num_expressions):
            meta = {}
            meta["video_id"] = video
            meta["exp"] = expressions[expression_list[i]]["question"]
            meta["ans"] = expressions[expression_list[i]]["answer"]
            meta["obj_id"] = int(expressions[expression_list[i]]["obj_id"])
            meta["q_type"] = expressions[expression_list[i]]["q_type"]
            meta["exp_id"] = expression_list[i]

            start = expressions[expression_list[i]]["action_start"]
            end = expressions[expression_list[i]]["action_end"]
            action_start = (time_str_to_seconds(start) - time_str_to_seconds(clip_start)) * 6 # fps=6
            action_end = (time_str_to_seconds(end) - time_str_to_seconds(clip_start)) * 6 - 1

            meta["action_start"] = action_start
            meta["action_end"] = action_end
            # meta["frame_dir"] = frame_start.zfill(4) + "_" + frame_end.zfill(4)
            metas.append(meta)
        meta = metas

        # 2. For each expression
        for i in range(num_expressions):
            video_id = meta[i]["video_id"]
            exp = meta[i]["exp"]
            exp_id = meta[i]["exp_id"]
            obj_id = meta[i]["obj_id"]
            q_type = meta[i]["q_type"]

            # action start and end is used to obtain gt masks in temporal dimension
            action_start = meta[i]["action_start"]  
            action_end = meta[i]["action_end"]

            if not os.path.exists(os.path.join(save_path_prefix, video_id, exp_id)):
                frame_dir = os.path.join(video_root, video_id, "images/")
                if not os.path.exists(frame_dir):
                    print("Missing frames: {}.".format(video_id))
                    continue
                raw_frames = os.listdir(frame_dir)  # all the frames
                sample_indices = np.linspace(0, len(raw_frames) - 1, num=20, dtype=int)  # uniformly sample 20 frames
                frames = [raw_frames[i] for i in sample_indices]
                video_len = len(frames)

                # read question for every video clip
                conv = conversation_lib.conv_templates[args.conv_type].copy()
                conv.messages = []

                prompt = exp + " Please respond with masks."
                prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
                if args.use_mm_start_end:
                    replace_token = (
                        DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                    )
                    prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

                conv.append_message(conv.roles[0], prompt)
                conv.append_message(conv.roles[1], "")
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                input_ids = input_ids.unsqueeze(0).cuda()

                images_list = []
                images_clip_list = []
                image_np_list = []
                original_size_list = []
                resize_list = []
                for t in range(video_len):
                    frame_id = frames[t]
                    image_path = os.path.join(frame_dir, frame_id)

                    image_np = cv2.imread(image_path)
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                    image_np_list.append(image_np)
                    original_size_list.append(image_np.shape[:2])

                    image_clip = (
                        clip_image_processor.preprocess(image_np, return_tensors="pt")[
                            "pixel_values"
                        ][0]
                        .unsqueeze(0)
                        .cuda()
                    )
                    if args.precision == "bf16":
                        image_clip = image_clip.bfloat16()
                    elif args.precision == "fp16":
                        image_clip = image_clip.half()
                    else:
                        image_clip = image_clip.float()

                    image = transform.apply_image(image_np)
                    resize_list.append(image.shape[:2])

                    image = (
                        preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
                        .unsqueeze(0)
                        .cuda()
                    )
                    if args.precision == "bf16":
                        image = image.bfloat16()
                    elif args.precision == "fp16":
                        image = image.half()
                    else:
                        image = image.float()

                    images_list.append(image)
                    images_clip_list.append(image_clip)
                
                image = torch.cat(images_list, dim=0).unsqueeze(0)
                image_clip = torch.cat(images_clip_list, dim=0).unsqueeze(0)

                input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                input_ids = input_ids.unsqueeze(0).cuda()

                output_ids, pred_masks = model.evaluate(
                    image_clip,
                    image,
                    input_ids,
                    resize_list,
                    original_size_list,
                    max_new_tokens=512,
                    tokenizer=tokenizer,
                )
                output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

                text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
                text_output = text_output.replace("\n", "").replace("  ", " ")
                # print("text_output: ", text_output)
                # print("Num masks: ", len(pred_masks))
                pred_mask_list = []
                for i, pred_mask in enumerate(pred_masks):
                    if pred_mask.shape[0] == 0:
                        continue

                    pred_mask = pred_mask.detach().cpu().numpy()[0]
                    pred_mask = pred_mask > 0
                    pred_mask_list.append(pred_mask)

                all_pred_masks = np.stack(pred_mask_list, axis=0)
                origin_h, origin_w = original_size_list[0]

                # load gt masks
                mask_dir = os.path.join(video_root, video_id, "masks/")
                gt_masks_list = []
                for index in sample_indices:
                    if action_start <= index <= action_end:
                        mask_id = "frame_" + str(index).zfill(6) + ".png"
                        mask_path = os.path.join(mask_dir, mask_id)
                        if os.path.exists(mask_path):
                            raw_mask = Image.open(mask_path).convert('P')
                        else:
                            raw_mask = np.zeros((origin_h, origin_w), dtype=np.int32)  # need to add frame index annotation in the meta file
                        raw_mask = np.array(raw_mask)
                        gt_mask = (raw_mask==obj_id).astype(np.float32)
                    else:
                        gt_mask = np.zeros((origin_h, origin_w), dtype=np.int32)

                    gt_masks_list.append(gt_mask) # list[mask]
                gt_masks = np.stack(gt_masks_list, axis=0)

                # calculate J & F
                j_metric = db_eval_iou(gt_masks, all_pred_masks)
                f_metric = db_eval_boundary(gt_masks, all_pred_masks)
                [JM, JR, JD] = db_statistics(j_metric)
                [FM, FR, FD] = db_statistics(f_metric)

                # print(video_id, JM, FM)
                JF = (JM + FM) / 2
                file.write(f'{video_id}, {exp_id}, {JF}, {JM}, {FM}\n')

                J_list.append(JM)
                F_list.append(FM)

                if q_type == "Causal":
                    causal_J_list.append(JM)
                    causal_F_list.append(FM)
                elif q_type == "Sequential":
                    sequential_J_list.append(JM)
                    sequential_F_list.append(FM)
                elif q_type == "Counterfactual":
                    counterfactual_J_list.append(JM)
                    counterfactual_F_list.append(FM)
                elif q_type == "Descriptive":
                    descriptive_J_list.append(JM)
                    descriptive_F_list.append(FM)

    final_J = np.mean(J_list)
    final_F = np.mean(F_list)
    final_JF = (final_J + final_F) / 2

    final_causal_J = np.mean(causal_J_list)
    final_causal_F = np.mean(causal_F_list)
    final_sequential_J = np.mean(sequential_J_list)
    final_sequential_F = np.mean(sequential_F_list)
    final_counterfactual_J = np.mean(counterfactual_J_list)
    final_counterfactual_F = np.mean(counterfactual_F_list)
    final_descriptive_J = np.mean(descriptive_J_list)
    final_descriptive_F = np.mean(descriptive_F_list)

    final_causal_JF = (final_causal_J + final_causal_F) / 2
    final_sequential_JF = (final_sequential_J + final_sequential_F) / 2
    final_counterfactual_JF = (final_counterfactual_J + final_counterfactual_F) / 2
    final_descriptive_JF = (final_descriptive_J + final_descriptive_F) / 2


    print(f"Final J (Jaccard Index): {final_J:.4f}")
    print(f"Final F (F-measure): {final_F:.4f}")
    print(f"Final JF (Average of J and F): {final_JF:.4f}\n")

    print(f"Final Causal J: {final_causal_J:.4f}")
    print(f"Final Causal F: {final_causal_F:.4f}")
    print(f"Final Causal JF (Average of Causal J and F): {final_causal_JF:.4f}\n")

    print(f"Final Sequential J: {final_sequential_J:.4f}")
    print(f"Final Sequential F: {final_sequential_F:.4f}")
    print(f"Final Sequential JF (Average of Sequential J and F): {final_sequential_JF:.4f}\n")

    print(f"Final Counterfactual J: {final_counterfactual_J:.4f}")
    print(f"Final Counterfactual F: {final_counterfactual_F:.4f}")
    print(f"Final Counterfactual JF (Average of Counterfactual J and F): {final_counterfactual_JF:.4f}\n")

    print(f"Final Descriptive J: {final_descriptive_J:.4f}")
    print(f"Final Descriptive F: {final_descriptive_F:.4f}")
    print(f"Final Descriptive JF (Average of Descriptive J and F): {final_descriptive_JF:.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
