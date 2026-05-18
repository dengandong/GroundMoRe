import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.segment_anything.utils.transforms import ResizeLongestSide

from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX)


def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    from model.llava import conversation as conversation_lib
    from model.llava.mm_utils import tokenizer_image_token

    # image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    loc_targets_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    video_id_list = []
    exp_id_list = []
    for sample in batch:
        if len(sample) == 12:
            (
                images,
                images_clip,
                conversations,
                masks,
                label,
                resize,
                questions,
                sampled_classes,
                video_id,
                exp_id,
                loc_targets,
                inference,
            ) = sample
        else:
            (
                images,
                images_clip,
                conversations,
                masks,
                label,
                resize,
                questions,
                sampled_classes,
                video_id,
                exp_id,
                inference,
            ) = sample
            loc_targets = (masks.flatten(1).sum(dim=1) > 0).float()
        # image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        loc_targets_list.append(loc_targets.float())
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)
        video_id_list.append(video_id)
        exp_id_list.append(exp_id)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        # "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "loc_targets_list": loc_targets_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
        "video_id_list": video_id_list,
        "exp_id_list": exp_id_list
    }


def _qwen_messages(question, answer=None):
    question = question.replace(DEFAULT_IMAGE_TOKEN, "").strip()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def qwen_collate_fn(
    batch,
    processor=None,
    tokenizer=None,
    model_max_length=1024,
    use_loc_token=True,
    local_rank=-1,
):
    if processor is None:
        raise ValueError("qwen_collate_fn requires a Qwen AutoProcessor.")
    if tokenizer is None:
        tokenizer = processor.tokenizer

    images_list = []
    images_clip_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    loc_targets_list = []
    qwen_images_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    video_id_list = []
    exp_id_list = []
    full_texts = []
    prompt_texts = []
    first_images = []
    frame_texts = []
    frame_images = []
    answer = "[SEG], [LOC]." if use_loc_token else "[SEG]."

    for sample in batch:
        if len(sample) == 12:
            (
                images,
                images_clip,
                conversations,
                masks,
                label,
                resize,
                questions,
                sampled_classes,
                video_id,
                exp_id,
                qwen_images,
                inference,
            ) = sample
            loc_targets = (masks.flatten(1).sum(dim=1) > 0).float()
        elif len(sample) == 13:
            (
                images,
                images_clip,
                conversations,
                masks,
                label,
                resize,
                questions,
                sampled_classes,
                video_id,
                exp_id,
                loc_targets,
                qwen_images,
                inference,
            ) = sample
        else:
            raise ValueError(
                "Qwen collate expects samples with qwen_images. Got sample "
                f"length {len(sample)}."
            )

        images_list.append(images)
        images_clip_list.append(images_clip)
        masks_list.append(masks.float())
        label_list.append(label)
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        loc_targets_list.append(loc_targets.float())
        qwen_images_list.append(qwen_images)
        cnt += len(questions)
        offset_list.append(cnt)
        inferences.append(inference)
        video_id_list.append(video_id)
        exp_id_list.append(exp_id)

        if len(qwen_images) != images.size(0):
            raise ValueError(
                f"Expected {images.size(0)} Qwen images, got {len(qwen_images)}."
            )
        for question in questions:
            full_texts.append(
                processor.apply_chat_template(
                    _qwen_messages(question, answer=answer),
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            prompt_texts.append(
                processor.apply_chat_template(
                    _qwen_messages(question, answer=None),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            first_images.append(qwen_images[0])
        for image in qwen_images:
            frame_texts.append(prompt_texts[-1])
            frame_images.append(image)

    full_inputs = processor(
        text=full_texts,
        images=first_images,
        padding=True,
        truncation=True,
        max_length=model_max_length,
        return_tensors="pt",
    )
    prompt_inputs = processor(
        text=prompt_texts,
        images=first_images,
        padding=True,
        truncation=True,
        max_length=model_max_length,
        return_tensors="pt",
    )
    frame_inputs = processor(
        text=frame_texts,
        images=frame_images,
        padding=True,
        truncation=True,
        max_length=model_max_length,
        return_tensors="pt",
    )

    input_ids = full_inputs["input_ids"]
    attention_masks = full_inputs["attention_mask"].bool()
    labels = input_ids.clone()
    labels[~attention_masks] = IGNORE_INDEX
    prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()
    for row_idx, prompt_len in enumerate(prompt_lengths):
        labels[row_idx, : int(prompt_len)] = IGNORE_INDEX

    return {
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": labels,
        "attention_masks": attention_masks,
        "pixel_values": frame_inputs["pixel_values"],
        "image_grid_thw": frame_inputs["image_grid_thw"],
        **(
            {"mm_token_type_ids": full_inputs["mm_token_type_ids"]}
            if "mm_token_type_ids" in full_inputs
            else {}
        ),
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "loc_targets_list": loc_targets_list,
        "inference": inferences[0],
        "conversation_list": full_texts,
        "video_id_list": video_id_list,
        "exp_id_list": exp_id_list,
    }


class VideoHybridDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 1024,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        dataset="video_seg||refer_video_seg||video_qa",
        sample_rate=[3, 1],
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refytvos",
        vqa_data="llava_instruct_150k",
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
        use_loc_token=True,
        num_frames=5,
        return_qwen_images=False,
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")
        if len(sample_rate) != len(self.datasets):
            raise ValueError(
                f"sample_rate has {len(sample_rate)} entries but dataset has "
                f"{len(self.datasets)} entries: {self.datasets}"
            )

        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "video_seg":
                pass
                # self.all_datasets.append(
                #     SemSegDataset(
                #         base_image_dir,
                #         tokenizer,
                #         vision_tower,
                #         samples_per_epoch,
                #         precision,
                #         image_size,
                #         num_classes_per_sample,
                #         exclude_val,
                #         sem_seg_data,
                #     )
                # )
            elif dataset == "refer_video_seg":
                from .refer_video_seg_dataset import YTVOSDataset

                self.all_datasets.append(
                    YTVOSDataset(
                        img_folder=os.path.join(base_image_dir, "refytvos", "train"),
                        ann_file=os.path.join(
                            base_image_dir,
                            "refytvos",
                            "meta_expressions",
                            "train",
                            "meta_expressions.json",
                        ),
                        tokenizer=tokenizer,
                        vision_tower=vision_tower,
                        # samples_per_epoch,
                        # precision,
                        image_size=image_size,
                        num_frames=num_frames,
                        max_skip=3,
                        return_masks=False,
                        use_loc_token=use_loc_token,
                        return_qwen_images=return_qwen_images,
                        # num_classes_per_sample,
                        # exclude_val,
                        # refer_seg_data,
                    )
                )
            elif dataset == "mevis":
                from .mevis_dataset import MeViSDataset

                self.all_datasets.append(
                    MeViSDataset(
                        img_folder=os.path.join(base_image_dir, "MeViSv2", "train"),
                        ann_file=os.path.join(
                            base_image_dir,
                            "MeViSv2",
                            "train",
                            "meta_expressions.json",
                        ),
                        tokenizer=tokenizer,
                        vision_tower=vision_tower,
                        image_size=image_size,
                        num_frames=num_frames,
                        max_skip=3,
                        return_masks=False,
                        use_loc_token=use_loc_token,
                        return_qwen_images=return_qwen_images,
                    )
                )
            elif dataset == "groundmore":
                from .groundmore_dataset import GroundMoReDataset

                self.all_datasets.append(
                    GroundMoReDataset(
                        root_dir=os.path.join(base_image_dir, "GroundMoRe"),
                        ann_file=os.path.join(
                            base_image_dir,
                            "GroundMoRe",
                            "trainval_v2.json",
                        ),
                        tokenizer=tokenizer,
                        vision_tower=vision_tower,
                        image_size=image_size,
                        num_frames=num_frames,
                        use_loc_token=use_loc_token,
                        return_qwen_images=return_qwen_images,
                    )
                )
            elif dataset == "video_qa":
                from .vqa_dataset import VQADataset

                self.all_datasets.append(
                    VQADataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        vqa_data,
                    )
                )
            elif dataset == "reason_seg":
                pass
                # self.all_datasets.append(
                #     ReasonSegDataset(
                #         base_image_dir,
                #         tokenizer,
                #         vision_tower,
                #         samples_per_epoch,
                #         precision,
                #         image_size,
                #         num_classes_per_sample,
                #         exclude_val,
                #         reason_seg_data,
                #         explanatory,
                #     )
                # )
            else:
                raise ValueError(f"Unsupported video dataset: {dataset}")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        inference = False
        return *data[0], inference  


class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
    ):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            images = glob.glob(
                os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg")
            )
            self.images = images
            self.data_type = "reason_seg"
        elif len(splits) == 3:
            from .refer import REFER

            ds, splitBy, split = splits
            refer_api = REFER(self.base_image_dir, ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        base_image_dir, "images/saiapr_tc-12", item["file_name"]
                    )
                elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
                    item["file_name"] = os.path.join(
                        base_image_dir,
                        "images/mscoco/images/train2014",
                        item["file_name"],
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

            img2refs = {}
            for ref in refs_val:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds
            self.data_type = "refer_seg"

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

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

    def __getitem__(self, idx):
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])

            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            is_sentence = False
        else:
            image_path = self.images[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            json_path = image_path.replace(".jpg", ".json")
            mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
            sampled_sents = [sampled_sents[0]]

        from model.llava import conversation as conversation_lib

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n {} Please output segmentation mask.".format(text),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            else:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n What is {} in this image? Please output segmentation mask.".format(
                        text
                    ),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            conversations.append(conv.get_prompt())
            i += 1

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if self.data_type == "refer_seg":
            masks = []
            for i, ann_id in enumerate(sampled_ann_ids):
                ann = annotations[ann_id]
                if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                    m = np.zeros((image_info["height"], image_info["width"], 1))
                else:
                    if type(ann["segmentation"][0]) == list:  # polygon
                        rle = mask.frPyObjects(
                            ann["segmentation"],
                            image_info["height"],
                            image_info["width"],
                        )
                    else:
                        rle = ann["segmentation"]
                        for i in range(len(rle)):
                            if not isinstance(rle[i]["counts"], bytes):
                                rle[i]["counts"] = rle[i]["counts"].encode()
                    m = mask.decode(rle)
                m = np.sum(
                    m, axis=2
                )  # sometimes there are multiple binary map (corresponding to multiple segs)
                m = m.astype(np.uint8)  # convert to np.uint8
                masks.append(m)
        else:
            masks = [mask_json]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            inference,
        )
