import argparse
import os
import sys
import json
import time
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.MoRA import MoRAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)

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
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--train_mask_decoder", action="store_true", default=False)
    parser.add_argument(
        "--seg_only",
        action="store_true",
        default=False,
        help="Disable [LOC] temporal localization and run with [SEG] only.",
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
    parser.add_argument("--video_root", default="./groundmore_videos/", type=str)
    parser.add_argument("--meta_file", default="groundmore_test.json", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--result_file", default="mora-zs-7b.txt", type=str)
    parser.add_argument("--num_frames", default=20, type=int)
    parser.add_argument("--max_new_tokens", default=32, type=int)
    parser.add_argument("--force_seg_token", action="store_true", default=False)
    parser.add_argument("--force_loc_token", action="store_true", default=False)
    parser.add_argument(
        "--use_loc_refine",
        action="store_true",
        default=False,
        help="Use predicted [LOC] temporal probabilities to zero masks outside the predicted temporal interval.",
    )
    parser.add_argument("--loc_threshold", default=0.5, type=float)
    parser.add_argument("--max_videos", default=None, type=int)
    parser.add_argument("--max_expressions", default=None, type=int)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Append to an existing result file and skip completed video/expression pairs.",
    )
    parser.add_argument(
        "--read_retries",
        default=3,
        type=int,
        help="Number of attempts for reading each frame before failing.",
    )
    return parser.parse_args(args)


def load_completed_results(result_file):
    completed = {}
    if not os.path.exists(result_file):
        return completed
    with open(result_file, "r") as f:
        for line in f:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 5:
                try:
                    completed[(parts[0], parts[1])] = (
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                    )
                except ValueError:
                    continue
    return completed


def imread_with_retry(image_path, retries=3, sleep_seconds=1.0):
    last_image = None
    for attempt in range(max(1, retries)):
        last_image = cv2.imread(image_path)
        if last_image is not None:
            return last_image
        if attempt + 1 < retries:
            time.sleep(sleep_seconds)
    return last_image


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
    tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.use_loc_token = not args.seg_only
    if args.use_loc_token:
        tokenizer.add_tokens("[LOC]")
        args.loc_token_idx = tokenizer("[LOC]", add_special_tokens=False).input_ids[0]
    else:
        args.loc_token_idx = None


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

    model = MoRAForCausalLM.from_pretrained(
        args.version,
        low_cpu_mem_usage=False,
        vision_tower=args.vision_tower,
        vision_pretrained=args.vision_pretrained,
        out_dim=args.out_dim,
        train_mask_decoder=args.train_mask_decoder,
        seg_token_idx=args.seg_token_idx,
        loc_token_idx=args.loc_token_idx,
        use_loc_token=args.use_loc_token,
        **kwargs,
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)
    if not hasattr(model.get_model(), "visual_model"):
        model.get_model().initialize_lisa_modules(model.get_model().config)

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
    video_root = args.video_root
    meta_file = args.meta_file
    save_path_prefix = args.output_dir
    os.makedirs(save_path_prefix, exist_ok=True)

    completed_results = load_completed_results(args.result_file) if args.resume else set()
    result_mode = "a" if args.resume else "w"
    file = open(args.result_file, result_mode)
    if args.resume:
        print(
            f"Resuming evaluation: loaded {len(completed_results)} completed "
            f"video/expression results from {args.result_file}",
            flush=True,
        )

    with open(meta_file, "r") as f:
        metadata = json.load(f)["videos"]

    video_list = list(metadata.keys())
    if args.max_videos is not None:
        video_list = video_list[: args.max_videos]

    # 1. For each video
    for video in tqdm(video_list):
        metas = [] # list[dict], length is number of expressions

        expressions = metadata[video]["questions"]   
        expression_list = list(expressions.keys()) 
        num_expressions = len(expression_list)
        if args.max_expressions is not None:
            num_expressions = min(num_expressions, args.max_expressions)

        clip_start = video[-9:].split("_")[0][:2] + ":" + video[-9:].split("_")[0][2:]
        clip_end = video[-9:].split("_")[1][:2] + ":" + video[-9:].split("_")[1][2:]

        # read all the anno meta
        for i in range(num_expressions):
            meta = {}
            meta["video_id"] = video
            meta["exp"] = expressions[expression_list[i]]["question"]
            meta["ans"] = expressions[expression_list[i]]["answer"]
            meta["obj_id"] = [
                int(obj_id.strip())
                for obj_id in str(expressions[expression_list[i]]["obj_id"]).split(",")
                if obj_id.strip()
            ]
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
            obj_ids = meta[i]["obj_id"]
            q_type = meta[i]["q_type"]
            if args.resume and (video_id, exp_id) in completed_results:
                _, JM, FM = completed_results[(video_id, exp_id)]
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
                continue

            # action start and end is used to obtain gt masks in temporal dimension
            action_start = meta[i]["action_start"]  
            action_end = meta[i]["action_end"]

            pred_dir = os.path.join(save_path_prefix, video_id, exp_id)
            should_run = args.overwrite or args.resume or not os.path.exists(pred_dir)
            if should_run:
                frame_dir = os.path.join(video_root, video_id, "images/")
                if not os.path.exists(frame_dir):
                    print("Missing frames: {}.".format(video_id))
                    continue
                raw_frames = sorted(
                    frame
                    for frame in os.listdir(frame_dir)
                    if frame.lower().endswith((".jpg", ".jpeg", ".png"))
                )
                if not raw_frames:
                    print("Missing readable frame files: {}.".format(video_id))
                    continue
                sample_indices = np.linspace(
                    0, len(raw_frames) - 1, num=args.num_frames, dtype=int
                )
                frames = [raw_frames[i] for i in sample_indices]
                video_len = len(frames)

                # read question for every video clip
                conv = conversation_lib.conv_templates[args.conv_type].copy()
                conv.messages = []

                prompt = exp + " Please output segmentation mask."
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

                    image_np = imread_with_retry(
                        image_path, retries=args.read_retries
                    )
                    if image_np is None:
                        raise FileNotFoundError(f"Failed to read frame: {image_path}")
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

                eval_outputs = model.evaluate(
                    image_clip,
                    image,
                    input_ids,
                    resize_list,
                    original_size_list,
                    max_new_tokens=args.max_new_tokens,
                    tokenizer=tokenizer,
                    force_seg_token=args.force_seg_token,
                    force_loc_token=args.force_loc_token,
                    return_loc=args.use_loc_refine,
                )
                if args.use_loc_refine:
                    output_ids, pred_masks, loc_probs = eval_outputs
                else:
                    output_ids, pred_masks = eval_outputs
                    loc_probs = None
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

                origin_h, origin_w = original_size_list[0]
                if not pred_mask_list:
                    print(
                        f"Warning: no predicted masks for {video_id}/{exp_id}; "
                        f"generated text: {text_output}",
                        flush=True,
                    )
                    all_pred_masks = np.zeros(
                        (video_len, origin_h, origin_w), dtype=bool
                    )
                else:
                    all_pred_masks = np.stack(pred_mask_list, axis=0)
                    if all_pred_masks.shape[0] < video_len:
                        pad = np.zeros(
                            (video_len - all_pred_masks.shape[0], origin_h, origin_w),
                            dtype=all_pred_masks.dtype,
                        )
                        all_pred_masks = np.concatenate([all_pred_masks, pad], axis=0)
                    elif all_pred_masks.shape[0] > video_len:
                        all_pred_masks = all_pred_masks[:video_len]
                    if loc_probs is not None:
                        loc_keep = (
                            loc_probs[0]
                            .detach()
                            .float()
                            .cpu()
                            .numpy()[:video_len]
                            >= args.loc_threshold
                        )
                        all_pred_masks[~loc_keep] = False

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
                        gt_mask = np.isin(raw_mask, obj_ids).astype(np.float32)
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
                file.flush()

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
