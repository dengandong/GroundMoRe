import argparse
import os
import shutil
import sys
import time
from functools import partial

import deepspeed
import torch
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from model.MoRA_Qwen3 import MoRAQwen3ForConditionalGeneration
from utils.video_dataset import VideoHybridDataset, ValDataset, qwen_collate_fn
from utils.utils import AverageMeter, ProgressMeter, dict_to_cuda


def parse_args(args):
    parser = argparse.ArgumentParser(description="MoRA-Qwen3 Model Training")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--version", default="Qwen/Qwen3-VL-7B-Instruct")
    parser.add_argument("--legacy_vision_tower", default="openai/clip-vit-large-patch14")
    parser.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=1024, type=int)
    parser.add_argument("--dataset", default="refer_video_seg||mevis")
    parser.add_argument("--sample_rates", default="1,1")
    parser.add_argument("--dataset_dir", default="./dataset")
    parser.add_argument("--log_base_dir", default="/data/exp/mora/")
    parser.add_argument("--exp_name", default="mora-qwen3-vl-7b-stage1")
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--steps_per_epoch", default=1000, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_frames", default=5, type=int)
    parser.add_argument("--grad_accumulation_steps", default=10, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--loc_loss_weight", default=1.0, type=float)
    parser.add_argument("--global_loc_loss_weight", default=1.0, type=float)
    parser.add_argument("--loc_mode", default="global", choices=["frame", "global"])
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj")
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=True)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--val_dataset", default="ReasonSeg|val")
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H")
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--init_from",
        default="",
        help=(
            "Initialize model weights from a DeepSpeed checkpoint without "
            "resuming optimizer, scheduler, or epoch counters."
        ),
    )
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--visualize_freq", default=100, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--seg_only", action="store_true", default=False)
    parser.add_argument("--auto_resume", action="store_true", default=False)
    parser.add_argument("--keep_ckpt_every", default=5, type=int)
    parser.add_argument("--pad_to_multiple_of", default=8, type=int)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--qwen_min_pixels", default=256 * 28 * 28, type=int)
    parser.add_argument("--qwen_max_pixels", default=1024 * 28 * 28, type=int)
    return parser.parse_args(args)


def find_linear_layers(model, lora_target_modules):
    cls = torch.nn.Linear
    lora_module_names = set()
    excluded = [
        "visual_model",
        "visual",
        "vision_tower",
        "text_hidden_fcs",
        "loc_temporal_head",
        "loc_global_context_proj",
        "loc_global_temporal_head",
    ]
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            and all(x not in name for x in excluded)
            and any(x in name for x in lora_target_modules)
        ):
            lora_module_names.add(name)
    return sorted(lora_module_names)


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")
    loc_losses = AverageMeter("LocLoss", ":.4f")
    loc_frame_losses = AverageMeter("LocFrameLoss", ":.4f")
    loc_global_losses = AverageMeter("LocGlobalLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            data_time,
            losses,
            ce_losses,
            mask_losses,
            loc_losses,
            loc_frame_losses,
            loc_global_losses,
            mask_bce_losses,
            mask_dice_losses,
        ],
        prefix=f"Epoch: [{epoch}]",
    )

    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for _ in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            image_keys = ("images", "images_clip", "pixel_values", "pixel_values_videos")
            for key in image_keys:
                if key not in input_dict:
                    continue
                if args.precision == "fp16":
                    input_dict[key] = input_dict[key].half()
                elif args.precision == "bf16":
                    input_dict[key] = input_dict[key].bfloat16()
                else:
                    input_dict[key] = input_dict[key].float()

            output_dict = model(**input_dict)
            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]
            loc_loss = output_dict.get("loc_loss", torch.zeros_like(loss))
            loc_frame_loss = output_dict.get("loc_frame_loss", torch.zeros_like(loss))
            loc_global_loss = output_dict.get("loc_global_loss", torch.zeros_like(loss))

            batch_n = input_dict["images"].size(0)
            losses.update(loss.item(), batch_n)
            ce_losses.update(ce_loss.item(), batch_n)
            mask_bce_losses.update(mask_bce_loss.item(), batch_n)
            mask_dice_losses.update(mask_dice_loss.item(), batch_n)
            mask_losses.update(mask_loss.item(), batch_n)
            loc_losses.update(loc_loss.item(), batch_n)
            loc_frame_losses.update(loc_frame_loss.item(), batch_n)
            loc_global_losses.update(loc_global_loss.item(), batch_n)
            model.backward(loss)
            model.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()
                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()
                loc_losses.all_reduce()
                loc_frame_losses.all_reduce()
                loc_global_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                cur_step = global_step + epoch * args.steps_per_epoch
                writer.add_scalar("train/loss", losses.avg, cur_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, cur_step)
                writer.add_scalar("train/mask_bce_loss", mask_bce_losses.avg, cur_step)
                writer.add_scalar("train/mask_dice_loss", mask_dice_losses.avg, cur_step)
                writer.add_scalar("train/mask_loss", mask_losses.avg, cur_step)
                writer.add_scalar("train/loc_loss", loc_losses.avg, cur_step)
                writer.add_scalar("train/loc_frame_loss", loc_frame_losses.avg, cur_step)
                writer.add_scalar("train/loc_global_loss", loc_global_losses.avg, cur_step)
                writer.add_scalar("metrics/total_secs_per_batch", batch_time.avg, cur_step)
                writer.add_scalar("metrics/data_secs_per_batch", data_time.avg, cur_step)

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()
            loc_losses.reset()
            loc_frame_losses.reset()
            loc_global_losses.reset()

    return train_iter


def validate(*_args, **_kwargs):
    raise NotImplementedError("MoRA-Qwen validation is not wired yet. Run with --no_eval.")


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

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
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.use_loc_token = not args.seg_only
    if args.use_loc_token:
        tokenizer.add_tokens("[LOC]", special_tokens=False)
        args.loc_token_idx = tokenizer("[LOC]", add_special_tokens=False).input_ids[0]
    else:
        args.loc_token_idx = None

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "loc_loss_weight": args.loc_loss_weight,
        "global_loc_loss_weight": args.global_loc_loss_weight,
        "loc_mode": args.loc_mode,
        "seg_token_idx": args.seg_token_idx,
        "loc_token_idx": args.loc_token_idx,
        "use_loc_token": args.use_loc_token,
        "vision_pretrained": args.vision_pretrained,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    model = MoRAQwen3ForConditionalGeneration.from_pretrained(
        args.version,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        **model_args,
    )
    # from_pretrained constructs custom modules inside HF's no-init context.
    # Rebuild all MoRA/SAM modules afterwards so their weights are real,
    # initialized tensors instead of meta/no-init placeholders.
    model.initialize_mora_modules()
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=args.pad_to_multiple_of)
    model.enable_input_require_grads()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    for name, param in model.named_parameters():
        if name.startswith("visual.") or ".visual." in name:
            param.requires_grad = False

    if args.lora_r > 0:
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        if args.local_rank == 0:
            print("LoRA target modules:", lora_target_modules)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    for name, param in model.named_parameters():
        if any(
            x in name
            for x in [
                "lm_head",
                "embed_tokens",
                "mask_decoder",
                "text_hidden_fcs",
                "loc_temporal_head",
                "loc_global",
            ]
        ):
            if args.local_rank == 0:
                print("trainable:", name, tuple(param.shape))
            param.requires_grad = True

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    train_dataset = VideoHybridDataset(
        args.dataset_dir,
        tokenizer,
        args.legacy_vision_tower,
        samples_per_epoch=args.batch_size
        * args.grad_accumulation_steps
        * args.steps_per_epoch
        * world_size,
        precision=args.precision,
        image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
        exclude_val=args.exclude_val,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        use_loc_token=args.use_loc_token,
        num_frames=args.num_frames,
        return_qwen_images=True,
    )

    if args.no_eval is False:
        val_dataset = ValDataset(
            args.dataset_dir,
            tokenizer,
            args.legacy_vision_tower,
            args.val_dataset,
            args.image_size,
        )
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
        )
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
                "torch_adam": True,
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {"enabled": args.precision == "fp16"},
        "bf16": {"enabled": args.precision == "bf16"},
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            qwen_collate_fn,
            processor=processor,
            tokenizer=tokenizer,
            model_max_length=args.model_max_length,
            use_loc_token=args.use_loc_token,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )

    if args.init_from and args.resume:
        raise ValueError("--init_from and --resume are mutually exclusive.")

    if args.init_from:
        model_engine.load_checkpoint(
            args.init_from,
            load_optimizer_states=False,
            load_lr_scheduler_states=False,
            load_module_strict=False,
            load_module_only=True,
        )
        print(f"initialized model weights from {args.init_from}")

    if args.auto_resume and len(args.resume) == 0 and len(args.init_from) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        print(f"resume training from {args.resume}, start from epoch {args.start_epoch}")

    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                qwen_collate_fn,
                processor=processor,
                tokenizer=tokenizer,
                model_max_length=args.model_max_length,
                use_loc_token=args.use_loc_token,
                local_rank=args.local_rank,
            ),
        )

    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0

    if args.eval_only:
        validate(val_loader, model_engine, 0, writer, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )

        is_best = False
        if args.no_eval is False:
            giou, ciou = validate(val_loader, model_engine, epoch, writer, args)
            is_best = giou > best_score
            best_score = max(giou, best_score)
            cur_ciou = ciou if is_best else cur_ciou

        if args.no_eval or is_best:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            if args.local_rank == 0:
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(
                            best_score, cur_ciou
                        ),
                    ),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)
            torch.distributed.barrier()
            if (
                args.local_rank == 0
                and args.keep_ckpt_every > 0
                and (epoch + 1) % args.keep_ckpt_every == 0
            ):
                keep_dir = os.path.join(args.log_dir, f"ckpt_model_epoch_{epoch + 1:03d}")
                if os.path.exists(keep_dir):
                    shutil.rmtree(keep_dir)
                shutil.copytree(save_dir, keep_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
