###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2023
###########################################################################
"""
MeViS data loader
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
from PIL import Image
import json
import numpy as np
import random

from transformers import CLIPImageProcessor

from pycocotools import mask as coco_mask

from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import ANSWER_LIST, ANSWER_LIST_WITH_LOC, SHORT_QUESTION_LIST


class MeViSDataset(Dataset):
    """
    A dataset class for the MeViS dataset which was first introduced in the paper:
    "MeViS: A Large-scale Benchmark for Video Segmentation with Motion Expressions"
    """

    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024

    def __init__(
        self,
        img_folder,
        ann_file,
        tokenizer=None,
        vision_tower=None,
        transforms=None,
        return_masks: bool = False,
        num_frames: int = 5,
        max_skip: int = 3,
        image_size: int = 1024,
        use_loc_token: bool = True,
        return_qwen_images: bool = False,
    ):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._transforms = transforms
        self.return_masks = return_masks  # not used
        self.num_frames = num_frames
        self.max_skip = max_skip
        self.img_size = image_size
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.return_qwen_images = return_qwen_images
        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST_WITH_LOC if use_loc_token else ANSWER_LIST

        # create video meta data
        self.prepare_metas()

        mask_json = os.path.join(str(self.img_folder), "mask_dict.json")
        print(f'Loading masks from {mask_json} ...')
        with open(mask_json) as fp:
            self.mask_dict = json.load(fp)

        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas))
        print('\n')

    def prepare_metas(self):
        # read expression data
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        self.videos = list(subset_expressions_by_video.keys())

        self.metas = []
        for vid in self.videos:
            # vid_meta = subset_metas_by_video[vid]
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)

            for exp_id, exp_dict in vid_data['expressions'].items():
                for frame_id in range(0, vid_len, self.num_frames):
                    meta = {}
                    meta['video'] = vid
                    meta['exp'] = exp_dict['exp']
                    meta['obj_id'] = [int(x) for x in exp_dict['obj_id']]
                    meta['anno_id'] = [str(x) for x in exp_dict['anno_id']]
                    meta['frames'] = vid_frames
                    meta['frame_id'] = frame_id
                    meta['category'] = 0
                    meta['exp_id'] = exp_id
                    self.metas.append(meta)

    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax  # y1, y2, x1, x2

    def __len__(self):
        return len(self.metas)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square SAM input."""
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        return F.pad(x, (0, padw, 0, padh))

    def __getitem__(self, idx):
        instance_check = False
        idx = random.randint(0, len(self.metas) - 1)
        while not instance_check:
            meta = self.metas[idx]  # dict

            video, exp, anno_id, category, frames, frame_id, exp_id = \
                meta['video'], meta['exp'], meta['anno_id'], meta['category'], meta['frames'], meta['frame_id'], meta['exp_id']
            # clean up the caption
            exp = " ".join(exp.lower().split())
            category_id = 0
            vid_len = len(frames)

            num_frames = self.num_frames
            # random sparse sample
            sample_indx = [frame_id]
            if self.num_frames != 1:
                # local sample
                sample_id_before = random.randint(1, 3)
                sample_id_after = random.randint(1, 3)
                local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
                sample_indx.extend(local_indx)

                # global sampling
                if num_frames > 3:
                    all_inds = list(range(vid_len))
                    global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                    global_n = num_frames - len(sample_indx)
                    if len(global_inds) > global_n:
                        select_id = random.sample(range(len(global_inds)), global_n)
                        for s_id in select_id:
                            sample_indx.append(global_inds[s_id])
                    elif vid_len >= global_n:  # sample long range global frames
                        select_id = random.sample(range(vid_len), global_n)
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])
                    else:
                        select_id = random.sample(range(vid_len), global_n - vid_len) + list(range(vid_len))
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])
            sample_indx.sort()

            # read frames and masks
            imgs, labels, boxes, masks, valid, img_clips, qwen_images = [], [], [], [], [], [], []
            for j in range(self.num_frames):
                frame_indx = sample_indx[j]
                frame_name = frames[frame_indx]
                img_path = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name + '.jpg')
                # mask_path = os.path.join(str(self.img_folder), 'Annotations', video, frame_name + '.png')
                img = Image.open(img_path).convert('RGB')
                if self.return_qwen_images:
                    qwen_images.append(img.copy())
                image_clip = self.clip_image_processor.preprocess(img, return_tensors="pt")[
                    "pixel_values"
                ][0]

                # h, w = img.shape
                mask = np.zeros(img.size[::-1], dtype=np.float32)
                for x in anno_id:
                    frm_anno = self.mask_dict[x][frame_indx]
                    if frm_anno is not None:
                        mask += coco_mask.decode(frm_anno)

                # create the target
                label = torch.tensor(category_id)

                if (mask > 0).any():
                    y1, y2, x1, x2 = self.bounding_box(mask)
                    box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                    valid.append(1)
                else:  # some frame didn't contain the instance
                    box = torch.tensor([0, 0, 0, 0]).to(torch.float)
                    valid.append(0)
                mask = torch.from_numpy(mask)

                # append
                img = np.array(img)
                img = self.transform.apply_image(img)
                resize = img.shape[:2]
                img = self.preprocess(torch.from_numpy(img).permute(2, 0, 1).contiguous())

                imgs.append(img)
                labels.append(label)
                masks.append(mask)
                boxes.append(box)
                img_clips.append(image_clip)

            # transform
            masks = torch.stack(masks, dim=0)
            target = {
                'frames_idx': torch.tensor(sample_indx),  # [T,]
                'masks': masks,  # [T, H, W]
                'valid': torch.tensor(valid),  # [T,]
                'caption': exp,
            }
            size_label = torch.ones(masks.size(1), masks.size(2)) * 255

            imgs = torch.stack(imgs, dim=0)  # [T, 3, H, W]
            img_clips = torch.stack(img_clips, dim=0)

            # FIXME: handle "valid", since some box may be removed due to random crop
            if torch.any(target['valid'] == 1):  # at leatst one instance
                instance_check = True
            else:
                idx = random.randint(0, self.__len__() - 1)

        sampled_classes = [exp]
        questions = []
        answers = []
        for text in sampled_classes:
            text = text.strip()
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))
            answers.append(random.choice(self.answer_list))

        conversations = []
        if not self.return_qwen_images:
            from model.llava import conversation as conversation_lib

            conv = conversation_lib.default_conversation.copy()
            for question, answer in zip(questions, answers):
                conv.messages = []
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], answer)
                conversations.append(conv.get_prompt())

        output = (
            imgs,
            img_clips,
            conversations,
            masks,
            size_label,
            resize,
            questions,
            sampled_classes,
            video,
            exp_id,
        )
        if self.return_qwen_images:
            output = output + (qwen_images,)
        return output
