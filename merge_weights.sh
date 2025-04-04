CUDA_VISIBLE_DEVICES="" python3 merge_lora_weights_and_save_hf_model.py \
  --version=liuhaotian/LLaVA-Lightning-7B-delta-v1-1 \
  --weight="lisa-7b/pytorch_model.bin" \
  --save_path="./LISA-7B"