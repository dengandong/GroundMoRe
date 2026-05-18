import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor

from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import ANSWER_LIST, ANSWER_LIST_WITH_LOC, DEFAULT_IMAGE_TOKEN


def time_str_to_seconds(time_str):
    parts = [int(x) for x in time_str.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Unsupported time format: {time_str}")


class GroundMoReDataset(Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024

    def __init__(
        self,
        root_dir,
        ann_file,
        tokenizer=None,
        vision_tower=None,
        num_frames=5,
        image_size=1024,
        use_loc_token=True,
        return_qwen_images=False,
    ):
        self.root_dir = root_dir
        self.video_root = os.path.join(root_dir, "groundmore_videos")
        self.ann_file = ann_file
        self.num_frames = num_frames
        self.img_size = image_size
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.answer_list = ANSWER_LIST_WITH_LOC if use_loc_token else ANSWER_LIST
        self.return_qwen_images = return_qwen_images

        self.metas = []
        with open(ann_file, "r") as f:
            videos = json.load(f)["videos"]
        for video_id, video_data in videos.items():
            image_dir = os.path.join(self.video_root, video_id, "images")
            mask_dir = os.path.join(self.video_root, video_id, "masks")
            if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
                continue

            frames = sorted(
                x
                for x in os.listdir(image_dir)
                if x.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            if not frames:
                continue

            clip_start = video_id[-9:].split("_")[0][:2] + ":" + video_id[-9:].split("_")[0][2:]
            clip_start_sec = time_str_to_seconds(clip_start)

            for exp_id, question in video_data["questions"].items():
                action_start = (
                    time_str_to_seconds(question["action_start"]) - clip_start_sec
                ) * 6
                action_end = (
                    time_str_to_seconds(question["action_end"]) - clip_start_sec
                ) * 6 - 1
                obj_ids = [
                    int(obj_id.strip())
                    for obj_id in str(question["obj_id"]).split(",")
                    if obj_id.strip()
                ]
                self.metas.append(
                    {
                        "video": video_id,
                        "frames": frames,
                        "question": " ".join(question["question"].strip().split()),
                        "answer": " ".join(question["answer"].strip().split()),
                        "obj_ids": obj_ids,
                        "exp_id": exp_id,
                        "action_start": float(action_start),
                        "action_end": float(action_end),
                    }
                )

        print(f"\nGroundMoRe video/question samples: {len(self.metas)}\n")

    def __len__(self):
        return len(self.metas)

    def preprocess(self, x):
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        return F.pad(x, (0, padw, 0, padh))

    def sample_frame_indices(self, vid_len, action_start, action_end):
        start = max(0, int(np.ceil(action_start)))
        end = min(vid_len - 1, int(np.floor(action_end)))
        if start <= end:
            action_indices = list(range(start, end + 1))
        else:
            action_indices = []

        outside_indices = [
            idx for idx in range(vid_len) if not (action_start <= idx <= action_end)
        ]

        if action_indices:
            num_pos = min(len(action_indices), max(1, self.num_frames // 2))
            pos = np.linspace(0, len(action_indices) - 1, num=num_pos, dtype=int)
            sampled = [action_indices[i] for i in pos.tolist()]
        else:
            sampled = []

        num_neg = self.num_frames - len(sampled)
        if outside_indices and num_neg > 0:
            neg = np.linspace(0, len(outside_indices) - 1, num=num_neg, dtype=int)
            sampled.extend(outside_indices[i] for i in neg.tolist())

        if len(sampled) < self.num_frames:
            fill = np.linspace(0, vid_len - 1, num=self.num_frames - len(sampled), dtype=int)
            sampled.extend(fill.tolist())

        return sorted(sampled[: self.num_frames])

    def __getitem__(self, idx):
        meta = self.metas[random.randint(0, len(self.metas) - 1)]
        frames = meta["frames"]
        vid_len = len(frames)
        sample_indices = self.sample_frame_indices(
            vid_len, meta["action_start"], meta["action_end"]
        )

        imgs, img_clips, masks, loc_targets = [], [], [], []
        qwen_images = []
        resize = None
        for frame_index in sample_indices:
            frame_name = frames[frame_index]
            image_path = os.path.join(
                self.video_root, meta["video"], "images", frame_name
            )
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
            if self.return_qwen_images:
                qwen_images.append(image)

            image_clip = self.clip_image_processor.preprocess(
                image, return_tensors="pt"
            )["pixel_values"][0]

            resized = self.transform.apply_image(image_np)
            resize = resized.shape[:2]
            img_tensor = self.preprocess(
                torch.from_numpy(resized).permute(2, 0, 1).contiguous()
            )

            in_action = meta["action_start"] <= frame_index <= meta["action_end"]
            loc_targets.append(1.0 if in_action else 0.0)

            if in_action:
                mask_name = "frame_" + str(frame_index).zfill(6) + ".png"
                mask_path = os.path.join(
                    self.video_root, meta["video"], "masks", mask_name
                )
                if os.path.exists(mask_path):
                    raw_mask = np.array(Image.open(mask_path).convert("P"))
                    mask = np.isin(raw_mask, meta["obj_ids"]).astype(np.float32)
                else:
                    mask = np.zeros(image_np.shape[:2], dtype=np.float32)
            else:
                mask = np.zeros(image_np.shape[:2], dtype=np.float32)

            imgs.append(img_tensor)
            img_clips.append(image_clip)
            masks.append(torch.from_numpy(mask))

        imgs = torch.stack(imgs, dim=0)
        img_clips = torch.stack(img_clips, dim=0)
        masks = torch.stack(masks, dim=0)
        size_label = torch.ones(masks.size(1), masks.size(2)) * 255
        loc_targets = torch.tensor(loc_targets, dtype=torch.float32)

        question = (
            DEFAULT_IMAGE_TOKEN
            + "\n"
            + meta["question"]
            + " Please output segmentation mask."
        )
        answer = random.choice(self.answer_list)

        if self.return_qwen_images:
            conversations = [question]
        else:
            from model.llava import conversation as conversation_lib

            conv = conversation_lib.default_conversation.copy()
            conv.messages = []
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], answer)
            conversations = [conv.get_prompt()]

        sample = (
            imgs,
            img_clips,
            conversations,
            masks,
            size_label,
            resize,
            [question],
            [meta["answer"]],
            meta["video"],
            meta["exp_id"],
            loc_targets,
        )
        if self.return_qwen_images:
            sample = (*sample, qwen_images)
        return sample
