import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from davis2017.metrics import db_eval_boundary, db_eval_iou
from davis2017.utils import db_statistics
from model.MoRA_Qwen3 import MoRAQwen3ForConditionalGeneration
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.video_dataset import _qwen_messages


def parse_args(args):
    parser = argparse.ArgumentParser(description="GroundMoRe evaluation for MoRA-Qwen")
    parser.add_argument("--version", default="Qwen/Qwen3-VL-7B-Instruct")
    parser.add_argument("--weight", required=True)
    parser.add_argument("--vision_pretrained", required=True)
    parser.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=4096, type=int)
    parser.add_argument("--video_root", default="groundmore_videos")
    parser.add_argument("--meta_file", default="groundmore_test.json")
    parser.add_argument("--output_dir", default="output_qwen")
    parser.add_argument("--result_file", default="mora-qwen-groundmore.txt")
    parser.add_argument("--num_frames", default=20, type=int)
    parser.add_argument("--loc_mode", default="global", choices=["frame", "global"])
    parser.add_argument("--loc_threshold", default=0.5, type=float)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj")
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--qwen_min_pixels", default=200704, type=int)
    parser.add_argument("--qwen_max_pixels", default=802816, type=int)
    parser.add_argument("--max_videos", default=None, type=int)
    parser.add_argument("--max_expressions", default=None, type=int)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--read_retries", default=3, type=int)
    return parser.parse_args(args)


def time_str_to_seconds(time_str):
    parts = [int(p) for p in time_str.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def imread_with_retry(image_path, retries=3, sleep_seconds=1.0):
    for attempt in range(max(1, retries)):
        image = cv2.imread(image_path)
        if image is not None:
            return image
        if attempt + 1 < retries:
            time.sleep(sleep_seconds)
    return None


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
):
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    return F.pad(x, (0, img_size - w, 0, img_size - h))


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


def find_linear_layers(model, target_names):
    excluded = [
        "visual_model",
        "visual",
        "vision_tower",
        "text_hidden_fcs",
        "loc_temporal_head",
        "loc_global_context_proj",
        "loc_global_temporal_head",
    ]
    names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, torch.nn.Linear)
            and all(x not in name for x in excluded)
            and any(x in name for x in target_names)
        ):
            names.add(name)
    return sorted(names)


def move_batch_to_cuda(batch, precision):
    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            batch[key] = value.cuda()
    for key in ("images", "pixel_values"):
        if key not in batch:
            continue
        if precision == "bf16":
            batch[key] = batch[key].bfloat16()
        elif precision == "fp16":
            batch[key] = batch[key].half()
        else:
            batch[key] = batch[key].float()
    return batch


def build_model(args):
    processor = transformers.AutoProcessor.from_pretrained(
        args.version,
        trust_remote_code=args.trust_remote_code,
        min_pixels=args.qwen_min_pixels,
        max_pixels=args.qwen_max_pixels,
    )
    tokenizer = processor.tokenizer
    tokenizer.model_max_length = args.model_max_length
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.add_tokens("[SEG]", special_tokens=False)
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    tokenizer.add_tokens("[LOC]", special_tokens=False)
    loc_token_idx = tokenizer("[LOC]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.float16

    model = MoRAQwen3ForConditionalGeneration.from_pretrained(
        args.version,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        train_mask_decoder=True,
        out_dim=args.out_dim,
        loc_mode=args.loc_mode,
        seg_token_idx=seg_token_idx,
        loc_token_idx=loc_token_idx,
        use_loc_token=True,
        vision_pretrained=args.vision_pretrained,
    )
    model.initialize_mora_modules()
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    if args.lora_r > 0:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=find_linear_layers(
                model, args.lora_target_modules.split(",")
            ),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    if os.path.isdir(args.weight):
        index_file = os.path.join(args.weight, "pytorch_model.bin.index.json")
        if not os.path.exists(index_file):
            raise FileNotFoundError(
                f"Expected sharded checkpoint index at {index_file}"
            )
        with open(index_file, "r") as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        missing = []
        unexpected = []
        for shard_file in shard_files:
            shard_path = os.path.join(args.weight, shard_file)
            shard_state = torch.load(shard_path, map_location="cpu")
            load_result = model.load_state_dict(
                shard_state, strict=False, assign=True
            )
            missing = list(load_result.missing_keys)
            unexpected.extend(load_result.unexpected_keys)
            del shard_state
    else:
        state_dict = torch.load(args.weight, map_location="cpu")
        if "module" in state_dict and isinstance(state_dict["module"], dict):
            state_dict = state_dict["module"]
        load_result = model.load_state_dict(state_dict, strict=False, assign=True)
        missing = list(load_result.missing_keys)
        unexpected = list(load_result.unexpected_keys)
    print(f"Loaded weight: {args.weight}", flush=True)
    print(f"Missing keys: {len(missing)}; unexpected keys: {len(unexpected)}", flush=True)
    if unexpected[:10]:
        print(f"First unexpected keys: {unexpected[:10]}", flush=True)

    if args.precision == "bf16":
        model = model.to(dtype=torch.bfloat16)
    elif args.precision == "fp16":
        model = model.to(dtype=torch.float16)
    else:
        model = model.to(dtype=torch.float32)
    model = model.cuda()
    model.eval()
    return model, processor


def build_batch(args, processor, exp, image_np_list, images_list, resize, origin_hw):
    question = exp + " Please output segmentation mask."
    answer = "[SEG], [LOC]."
    full_text = processor.apply_chat_template(
        _qwen_messages(question, answer=answer),
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_text = processor.apply_chat_template(
        _qwen_messages(question, answer=None),
        tokenize=False,
        add_generation_prompt=True,
    )
    qwen_images = [Image.fromarray(image_np) for image_np in image_np_list]
    first_image = qwen_images[0]
    full_inputs = processor(
        text=[full_text],
        images=[first_image],
        padding=True,
        truncation=True,
        max_length=args.model_max_length,
        return_tensors="pt",
    )
    frame_inputs = processor(
        text=[prompt_text] * len(qwen_images),
        images=qwen_images,
        padding=True,
        truncation=True,
        max_length=args.model_max_length,
        return_tensors="pt",
    )
    h, w = origin_hw
    masks = torch.zeros((len(qwen_images), h, w), dtype=torch.float32)
    label = torch.ones((h, w), dtype=torch.float32) * 255
    batch = {
        "images": torch.stack(images_list, dim=0).unsqueeze(0),
        "input_ids": full_inputs["input_ids"],
        "labels": full_inputs["input_ids"].clone(),
        "attention_masks": full_inputs["attention_mask"].bool(),
        "pixel_values": frame_inputs["pixel_values"],
        "image_grid_thw": frame_inputs["image_grid_thw"],
        "masks_list": [masks.cuda()],
        "label_list": [label.cuda()],
        "resize_list": [resize],
        "offset": torch.LongTensor([0, 1]),
        "questions_list": [[question]],
        "loc_targets_list": [torch.zeros(len(qwen_images), dtype=torch.float32).cuda()],
        "inference": True,
    }
    if "mm_token_type_ids" in full_inputs:
        batch["mm_token_type_ids"] = full_inputs["mm_token_type_ids"]
    return move_batch_to_cuda(batch, args.precision)


def update_metrics(q_type, jm, fm, metrics):
    metrics["overall"][0].append(jm)
    metrics["overall"][1].append(fm)
    if q_type in metrics:
        metrics[q_type][0].append(jm)
        metrics[q_type][1].append(fm)


def print_summary(metrics):
    labels = [
        ("overall", "Final"),
        ("Causal", "Final Causal"),
        ("Sequential", "Final Sequential"),
        ("Counterfactual", "Final Counterfactual"),
        ("Descriptive", "Final Descriptive"),
    ]
    for key, label in labels:
        j = float(np.mean(metrics[key][0])) if metrics[key][0] else float("nan")
        f = float(np.mean(metrics[key][1])) if metrics[key][1] else float("nan")
        jf = (j + f) / 2
        if key == "overall":
            print(f"{label} J (Jaccard Index): {j:.4f}")
            print(f"{label} F (F-measure): {f:.4f}")
            print(f"{label} JF (Average of J and F): {jf:.4f}\n")
        else:
            print(f"{label} J: {j:.4f}")
            print(f"{label} F: {f:.4f}")
            print(f"{label} JF (Average of {key} J and F): {jf:.4f}\n")


def main(argv):
    args = parse_args(argv)
    os.makedirs(os.path.dirname(args.result_file) or ".", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.overwrite and os.path.exists(args.result_file):
        os.remove(args.result_file)

    model, processor = build_model(args)
    transform = ResizeLongestSide(args.image_size)
    completed = load_completed_results(args.result_file) if args.resume else {}
    mode = "a" if args.resume else "w"
    result_f = open(args.result_file, mode)

    with open(args.meta_file, "r") as f:
        metadata = json.load(f)["videos"]
    video_list = list(metadata.keys())
    if args.max_videos is not None:
        video_list = video_list[: args.max_videos]

    metrics = {
        "overall": ([], []),
        "Causal": ([], []),
        "Sequential": ([], []),
        "Counterfactual": ([], []),
        "Descriptive": ([], []),
    }
    if completed:
        print(f"Resuming from {len(completed)} completed results.", flush=True)

    with torch.inference_mode():
        for video in tqdm(video_list):
            expressions = metadata[video]["questions"]
            expression_ids = list(expressions.keys())
            if args.max_expressions is not None:
                expression_ids = expression_ids[: args.max_expressions]

            clip_start = video[-9:].split("_")[0][:2] + ":" + video[-9:].split("_")[0][2:]
            frame_dir = os.path.join(args.video_root, video, "images")
            raw_frames = sorted(
                frame
                for frame in os.listdir(frame_dir)
                if frame.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            sample_indices = np.linspace(
                0, len(raw_frames) - 1, num=args.num_frames, dtype=int
            )
            sampled_frames = [raw_frames[idx] for idx in sample_indices]

            image_np_list = []
            images_list = []
            resize = None
            origin_hw = None
            for frame_name in sampled_frames:
                image_path = os.path.join(frame_dir, frame_name)
                image_np = imread_with_retry(image_path, args.read_retries)
                if image_np is None:
                    raise FileNotFoundError(f"Failed to read frame: {image_path}")
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                image_np_list.append(image_np)
                if origin_hw is None:
                    origin_hw = image_np.shape[:2]
                resized_np = transform.apply_image(image_np)
                if resize is None:
                    resize = resized_np.shape[:2]
                image = preprocess(
                    torch.from_numpy(resized_np).permute(2, 0, 1).contiguous(),
                    img_size=args.image_size,
                )
                images_list.append(image)

            for exp_id in expression_ids:
                item = expressions[exp_id]
                if args.resume and (video, exp_id) in completed:
                    _, jm, fm = completed[(video, exp_id)]
                    update_metrics(item["q_type"], jm, fm, metrics)
                    continue

                start = item["action_start"]
                end = item["action_end"]
                action_start = (
                    time_str_to_seconds(start) - time_str_to_seconds(clip_start)
                ) * 6
                action_end = (
                    time_str_to_seconds(end) - time_str_to_seconds(clip_start)
                ) * 6 - 1
                obj_ids = [
                    int(obj_id.strip())
                    for obj_id in str(item["obj_id"]).split(",")
                    if obj_id.strip()
                ]
                batch = build_batch(
                    args,
                    processor,
                    item["question"],
                    image_np_list,
                    images_list,
                    resize,
                    origin_hw,
                )
                output = model(**batch)
                pred_mask = output["pred_masks"][0].detach().float().cpu().numpy() > 0
                pred_mask = pred_mask[: args.num_frames]
                if pred_mask.shape[0] < args.num_frames:
                    pad = np.zeros(
                        (args.num_frames - pred_mask.shape[0], *origin_hw), dtype=bool
                    )
                    pred_mask = np.concatenate([pred_mask, pad], axis=0)
                if "loc_probs" in output:
                    loc_keep = (
                        output["loc_probs"][0]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()[: args.num_frames]
                        >= args.loc_threshold
                    )
                    pred_mask[~loc_keep] = False

                gt_masks = []
                mask_dir = os.path.join(args.video_root, video, "masks")
                for frame_idx in sample_indices:
                    if action_start <= frame_idx <= action_end:
                        mask_path = os.path.join(
                            mask_dir, "frame_" + str(frame_idx).zfill(6) + ".png"
                        )
                        if os.path.exists(mask_path):
                            raw_mask = np.array(Image.open(mask_path).convert("P"))
                        else:
                            raw_mask = np.zeros(origin_hw, dtype=np.int32)
                        gt_mask = np.isin(raw_mask, obj_ids).astype(np.float32)
                    else:
                        gt_mask = np.zeros(origin_hw, dtype=np.float32)
                    gt_masks.append(gt_mask)
                gt_masks = np.stack(gt_masks, axis=0)

                j_metric = db_eval_iou(gt_masks, pred_mask)
                f_metric = db_eval_boundary(gt_masks, pred_mask)
                jm = db_statistics(j_metric)[0]
                fm = db_statistics(f_metric)[0]
                jf = (jm + fm) / 2
                result_f.write(f"{video}, {exp_id}, {jf}, {jm}, {fm}\n")
                result_f.flush()
                update_metrics(item["q_type"], jm, fm, metrics)

    result_f.close()
    print_summary(metrics)


if __name__ == "__main__":
    main(sys.argv[1:])
