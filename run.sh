#!/bin/bash

deepspeed --master_port=24999 --include=localhost:0,1 train_ds.py \
--version="xinlai/LISA-7B-v1" \
--dataset_dir=None \
--vision_pretrained="./pretrain_weights/sam_vit_h_4b8939.pth" \
--dataset="refer_video_seg" \
--sample_rates="1" \
--conv_type="llava_llama_2" \
--exp_name="mora-lisa7b-zs-training" \
--log_base_dir="" \
--steps_per_epoch 500 \
--epochs 20 \
--auto_resume \
