"""
Ref-YoutubeVOS data loader
"""
import os
import sys
from pathlib import Path

import torch
# from torch.autograd.grad_mode import F
import torch.nn.functional as F
from torch.utils.data import Dataset
from .transforms_video import Compose, Check, ToTensor, Normalize, RandomHorizontalFlip, \
    PhotometricDistort, RandomSelect, RandomResize, RandomSizeCrop
from transformers import CLIPImageProcessor


from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide


import os
from PIL import Image
import json
import numpy as np
import random

import cv2

from .categories import ytvos_category_dict as category_dict
from .utils import ANSWER_LIST, SHORT_QUESTION_LIST


class YTVOSDataset(Dataset):
    """
    A dataset class for the Refer-Youtube-VOS dataset which was first introduced in the paper:
    "URVOS: Unified Referring Video Object Segmentation Network with a Large-Scale Benchmark"
    (see https://link.springer.com/content/pdf/10.1007/978-3-030-58555-6_13.pdf).
    The original release of the dataset contained both 'first-frame' and 'full-video' expressions. However, the first
    dataset is not publicly available anymore as now only the harder 'full-video' subset is available to download
    through the Youtube-VOS referring video object segmentation competition page at:
    https://competitions.codalab.org/competitions/29139
    Furthermore, for the competition the subset's original validation set, which consists of 507 videos, was split into
    two competition 'validation' & 'test' subsets, consisting of 202 and 305 videos respectively. Evaluation can
    currently only be done on the competition 'validation' subset using the competition's server, as
    annotations were publicly released only for the 'train' subset of the competition.

    """
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024

    def __init__(self, 
                 img_folder: Path, 
                 ann_file: Path, 
                 tokenizer,
                 vision_tower,
                 return_masks: bool, 
                 num_frames: int, 
                 max_skip: int,
                 image_size: int = 1024):
        
        self.img_folder = img_folder     
        self.ann_file = ann_file         
        self.return_masks = return_masks # not used
        self.num_frames = num_frames     
        self.max_skip = max_skip
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        # create video meta data
        self.prepare_metas()       

        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas))  
        print('\n')    

    # def prepare_metas(self):
    #     # read object information
    #     with open(os.path.join(str(self.img_folder), 'meta.json'), 'r') as f:
    #         subset_metas_by_video = json.load(f)['videos']
        
    #     # read expression data
    #     with open(str(self.ann_file), 'r') as f:
    #         subset_expressions_by_video = json.load(f)['videos']
    #     self.videos = list(subset_expressions_by_video.keys())

    #     self.metas = []
    #     for vid in self.videos:
    #         vid_meta = subset_metas_by_video[vid]
    #         vid_data = subset_expressions_by_video[vid]
    #         vid_frames = sorted(vid_data['frames'])
    #         vid_len = len(vid_frames)
    #         for exp_id, exp_dict in vid_data['expressions'].items():
    #             for frame_id in range(0, vid_len, self.num_frames):
    #                 meta = {}
    #                 meta['video'] = vid
    #                 meta['exp'] = exp_dict['exp']
    #                 meta['obj_id'] = int(exp_dict['obj_id'])
    #                 meta['frames'] = vid_frames
    #                 meta['frame_id'] = frame_id
    #                 # get object category
    #                 obj_id = exp_dict['obj_id']
    #                 meta['category'] = vid_meta['objects'][obj_id]['category']
    #                 meta['exp_id'] = exp_id
    #                 self.metas.append(meta)

    def prepare_metas(self):

        def create_meta_dict(expressions_dict):
            # Initialize a dictionary to store meta dicts for each obj_id
            meta_dicts = {}
            
            # Loop through each expression to populate meta dicts
            for exp in expressions_dict.values():
                obj_id = exp["obj_id"]
                if obj_id not in meta_dicts:
                    # Initialize a new meta dict for this obj_id
                    meta_dicts[obj_id] = {"obj_id": obj_id, "exp": []}
                # Append the expression to the list of expressions for this obj_id
                meta_dicts[obj_id]["exp"].append(exp["exp"])
            
            return meta_dicts
        
        # read object information
        with open(os.path.join(str(self.img_folder), 'meta.json'), 'r') as f:
            subset_metas_by_video = json.load(f)['videos']
        
        # read expression data
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        self.videos = list(subset_expressions_by_video.keys())

        self.metas = []
        for vid in self.videos:
            vid_meta = subset_metas_by_video[vid]
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)

            num_obj = len(vid_data['expressions'].keys()) // 2

            metas = create_meta_dict(vid_data['expressions'])
            for obj_id in metas.keys():
                meta = {}
                meta['video'] = vid
                meta['frames'] = vid_frames
                meta['category'] = vid_meta['objects'][obj_id]['category']
                meta["obj_id"] = obj_id
                meta["exp"] = metas[obj_id]["exp"]
                self.metas.append(meta)

    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax # y1, y2, x1, x2 
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
        
    def __len__(self):
        return len(self.metas)
        
    def __getitem__(self, idx):
        instance_check = False
        idx = random.randint(0, len(self.metas) - 1)
        while not instance_check:
            meta = self.metas[idx]  # dict
            # print("meta: ", meta)
            video, exps, obj_id, category, frames, frame_id, exp_id = \
                        meta['video'], meta['exp'], int(meta['obj_id']), meta['category'], meta['frames'], 0, 0
            # clean up the caption
            exps = [" ".join(exp.lower().split()) for exp in exps]
            # exp = " ".join(exp.lower().split())
            category_id = category_dict[category]
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
                    elif vid_len >=global_n:  # sample long range global frames
                        select_id = random.sample(range(vid_len), global_n)
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])
                    else:
                        select_id = random.sample(range(vid_len), global_n - vid_len) + list(range(vid_len))           
                        for s_id in select_id:                                                                   
                            sample_indx.append(all_inds[s_id])
            sample_indx.sort()

            # read frames and masks
            imgs, labels, boxes, masks, valid, img_clips = [], [], [], [], [], []
            random_id = int(torch.randint(low=0, high=10000, size=(1,))[0])
            for j in range(self.num_frames):
                frame_indx = sample_indx[j]
                frame_name = frames[frame_indx]
                img_path = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name + '.jpg')
                mask_path = os.path.join(str(self.img_folder), 'Annotations', video, frame_name + '.png')
                img = Image.open(img_path).convert('RGB')
                mask = Image.open(mask_path).convert('P')

                # preprocess image for clip
                image_clip = self.clip_image_processor.preprocess(img, return_tensors="pt")[
                    "pixel_values"
                ][0]
                img = np.array(img)
                img = self.transform.apply_image(img)  # preprocess image for sam

                # cv2.imwrite("examples/img_{}_{}.png".format(random_id, j), np.array(img))

                resize = np.array(img).shape[:2]

                img = self.preprocess(torch.from_numpy(img).permute(2, 0, 1).contiguous())

                # create the target
                label =  torch.tensor(category_id) 
                mask = np.array(mask)

                mask = (mask==obj_id).astype(np.float32) # 0,1 binary
                if (mask > 0).any():
                    y1, y2, x1, x2 = self.bounding_box(mask)
                    box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                    valid.append(1)
                else: # some frame didn't contain the instance
                    box = torch.tensor([0, 0, 0, 0]).to(torch.float) 
                    valid.append(0)
                mask = torch.from_numpy(mask)

                # append
                imgs.append(img)
                labels.append(label)
                masks.append(mask)
                boxes.append(box)
                img_clips.append(image_clip)

            masks = torch.stack(masks, dim=0) 
            size_label = torch.ones(masks.size(1), masks.size(2)) * 255

            target = {
                'frames_idx': torch.tensor(sample_indx), # [T,]
                'labels': labels,                        # [T,]
                # 'boxes': boxes,                          # [T, 4], xyxy
                'masks': masks,                          # [T, H, W]
                'valid': torch.tensor(valid),            # [T,]
                'caption': exps,
                # 'orig_size': torch.as_tensor([int(h), int(w)]), 
                # 'size': torch.as_tensor([int(h), int(w)])
            }

            imgs = torch.stack(imgs, dim=0) # [T, 3, H, W]
            img_clips = torch.stack(img_clips, dim=0) # [T, **]

            # FIXME: handle "valid", since some box may be removed due to random crop
            if torch.any(target['valid'] == 1):  # at leatst one instance
                instance_check = True
            else:
                idx = random.randint(0, self.__len__() - 1)

            sampled_classes = exps  # ?? is it?

            questions = []
            answers = []
            for text in sampled_classes:
                text = text.strip()
                assert len(text.split("||")) == 1
                question_template = random.choice(self.short_question_list)
                questions.append(question_template.format(class_name=text.lower()))
                answers.append(random.choice(self.answer_list))

            conversations = []
            conv = conversation_lib.default_conversation.copy()

            i = 0
            while i < len(questions):
                conv.messages = []
                conv.append_message(conv.roles[0], questions[i])
                conv.append_message(conv.roles[1], answers[i])
                conversations.append(conv.get_prompt())
                i += 1

        return (imgs, 
                img_clips,
                conversations,  # only one for each video?
                masks,
                size_label,
                resize,
                questions,
                sampled_classes,
                video,
                exp_id
                )


class YTVOSValDataset(Dataset):
    """
    A dataset class for the Refer-Youtube-VOS dataset which was first introduced in the paper:
    "URVOS: Unified Referring Video Object Segmentation Network with a Large-Scale Benchmark"
    (see https://link.springer.com/content/pdf/10.1007/978-3-030-58555-6_13.pdf).
    The original release of the dataset contained both 'first-frame' and 'full-video' expressions. However, the first
    dataset is not publicly available anymore as now only the harder 'full-video' subset is available to download
    through the Youtube-VOS referring video object segmentation competition page at:
    https://competitions.codalab.org/competitions/29139
    Furthermore, for the competition the subset's original validation set, which consists of 507 videos, was split into
    two competition 'validation' & 'test' subsets, consisting of 202 and 305 videos respectively. Evaluation can
    currently only be done on the competition 'validation' subset using the competition's server, as
    annotations were publicly released only for the 'train' subset of the competition.

    """
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024

    def __init__(self, 
                 img_folder: Path, 
                 ann_file: Path, 
                 tokenizer,
                 vision_tower,
                 image_size: int = 1024):
        
        self.img_folder = img_folder     
        self.ann_file = ann_file         
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        # create video meta data
        self.prepare_metas()       

        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas))  
        print('\n')    

    def prepare_metas(self):
        
        # read expression data
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        self.videos = list(subset_expressions_by_video.keys())

        self.metas = []
        for vid in self.videos:
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            for exp_id, exp_dict in vid_data['expressions'].items():
                for frame_id in range(0, vid_len):
                    meta = {}
                    meta['video'] = vid
                    meta['exp'] = exp_dict['exp']
                    meta['frames'] = vid_frames
                    meta['frame_id'] = frame_id
                    meta['exp_id'] = exp_id
                    self.metas.append(meta)

    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax # y1, y2, x1, x2 
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
        
    def __len__(self):
        return len(self.metas)
        
    def __getitem__(self, idx):
        # instance_check = False
        # while not instance_check:
        meta = self.metas[idx]  # dict
        # print("meta: ", meta)
        video, exp, frames, frame_id, exp_id = \
                    meta['video'], meta['exp'], meta['frames'], meta['frame_id'], meta['exp_id']
        # clean up the caption
        exp = " ".join(exp.lower().split())
        vid_len = len(frames)

        num_frames = vid_len
        num_frames = 3
        # random sparse sample
        sample_indx = list(range(vid_len))
        # if num_frames != 1:
        #     # local sample
        #     sample_id_before = random.randint(1, 3)
        #     sample_id_after = random.randint(1, 3)
        #     local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
        #     sample_indx.extend(local_indx)

        #     # global sampling
        #     if num_frames > 3:
        #         all_inds = list(range(vid_len))
        #         global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
        #         global_n = num_frames - len(sample_indx)
        #         if len(global_inds) > global_n:
        #             select_id = random.sample(range(len(global_inds)), global_n)
        #             for s_id in select_id:
        #                 sample_indx.append(global_inds[s_id])
        #         elif vid_len >=global_n:  # sample long range global frames
        #             select_id = random.sample(range(vid_len), global_n)
        #             for s_id in select_id:
        #                 sample_indx.append(all_inds[s_id])
        #         else:
        #             select_id = random.sample(range(vid_len), global_n - vid_len) + list(range(vid_len))           
        #             for s_id in select_id:                                                                   
        #                 sample_indx.append(all_inds[s_id])
        # sample_indx.sort()

        # read frames and masks
        imgs, labels, boxes, valid, img_clips = [], [], [], [], []
        random_id = int(torch.randint(low=0, high=10000, size=(1,))[0])
        for j in range(num_frames):
            frame_indx = sample_indx[j]
            frame_name = frames[frame_indx]
            img_path = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name + '.jpg')
            img = Image.open(img_path).convert('RGB')

            # preprocess image for clip
            image_clip = self.clip_image_processor.preprocess(img, return_tensors="pt")[
                "pixel_values"
            ][0]
            img = np.array(img)
            size_label = torch.ones(img.shape[0], img.shape[1]) * 255
            img = self.transform.apply_image(img)  # preprocess image for sam

            resize = np.array(img).shape[:2]

            img = self.preprocess(torch.from_numpy(img).permute(2, 0, 1).contiguous())

            # create the target

            # append
            imgs.append(img)
            img_clips.append(image_clip)

        imgs = torch.stack(imgs, dim=0) # [T, 3, H, W]
        img_clips = torch.stack(img_clips, dim=0) # [T, **]

        sampled_classes = [exp]  # ?? is it?

        questions = []
        answers = []
        for text in sampled_classes:
            text = text.strip()
            assert len(text.split("||")) == 1
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))
            answers.append(random.choice(self.answer_list))

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1
        inference = False
        
        return (imgs, 
                img_clips,
                conversations,
                torch.tensor([0.]),
                size_label,
                resize,
                questions,
                sampled_classes,
                video,
                exp_id,
                inference
                )
