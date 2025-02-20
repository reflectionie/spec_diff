#!/usr/bin/env python
# coding=utf-8
""" 
python train_hidden_state_diffuser_v2_scheduler.py \
  --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct \
  --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data \
  --checkpointing_steps 100 --learning_rate 1e-6 --train_batch_size 32768 --warmup_steps 1000

 python train_hidden_state_diffuser_v2_scheduler.py   --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct   --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data   --checkpointing_steps 100  --train_batch_size 32768 --warmup_steps 1000
"""

import argparse
import logging
import math
import os
from pathlib import Path
import json
from safetensors import safe_open
from transformers import AutoConfig
##### 新增导入：学习率调度器
from transformers import get_linear_schedule_with_warmup

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm

import wandb

from diffusers import DDPMScheduler
from concurrent.futures import ThreadPoolExecutor

# 导入我们定义的条件扩散模型
# from my_hidden_state_diffusion import HiddenStateDiffusionModel, HiddenStateDiffusionOutput
# from my_hidden_state_diffusion_v2 import HiddenStateDiffusionModel, HiddenStateDiffusionOutput
from my_hidden_state_diffusion_v1_concat import HiddenStateDiffusionModel, HiddenStateDiffusionOutput

logger = get_logger(__name__, log_level="INFO")


###############################################################################
# 新版数据集：同一个文件里同时包含 "draft_hidden" 和 "hidden_state"
###############################################################################
class ChainedHiddenStateBatchDataset(Dataset):
    """
    该数据集将多个文件中的 hidden state 按顺序拼接成一个长序列，
    然后每次返回一个连续片段，该片段的长度等于 batch_size（即 chunk_size）。
    """

    def __init__(self, paths, chunk_size, num_workers=40):
        super().__init__()
        self.paths = paths
        self.chunk_size = chunk_size

        # 预先计算每个文件中的 hidden state 数量，以及累积长度
        self.file_lengths = []
        self.cumulative_lengths = []
        cumulative = 0

        print("Counting data total_length with multi-threading...")

        def get_file_length(p):
            data = torch.load(p, map_location="cpu")
            draft = data["draft_hidden"].squeeze(0)  # [seq_len, hidden_dim]
            return draft.shape[0]  # draft_hidden 与 hidden_state 长度应相同

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            file_lengths = list(tqdm(executor.map(get_file_length, self.paths), total=len(self.paths), ncols=80))
        
        for length in file_lengths:
            self.file_lengths.append(length)
            cumulative += length
            self.cumulative_lengths.append(cumulative)
        self.total_length = cumulative

        # 按照 chunk_size，计算总共能构成多少个完整 batch
        self.num_batches = self.total_length // self.chunk_size

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        start = index * self.chunk_size
        end = start + self.chunk_size

        draft_chunks = []
        gt_chunks = []
        current_global = start

        while current_global < end:
            file_idx = self._find_file_index(current_global)
            file_start_global = self.cumulative_lengths[file_idx - 1] if file_idx > 0 else 0
            local_start = current_global - file_start_global

            available = self.file_lengths[file_idx] - local_start
            needed = end - current_global
            take = min(available, needed)

            loaded = torch.load(self.paths[file_idx])
            draft_data = loaded["draft_hidden"].squeeze(0)  # [seq_len, hidden_dim]
            gt_data = loaded["hidden_state"].squeeze(0)     # [seq_len, hidden_dim]

            draft_chunks.append(draft_data[local_start: local_start + take])
            gt_chunks.append(gt_data[local_start: local_start + take])

            current_global += take

        draft_batch = torch.cat(draft_chunks, dim=0)
        gt_batch = torch.cat(gt_chunks, dim=0)

        assert draft_batch.shape[0] == self.chunk_size
        assert gt_batch.shape[0] == self.chunk_size

        return {"draft": draft_batch, "gt": gt_batch}

    def _find_file_index(self, global_index):
        low, high = 0, len(self.cumulative_lengths) - 1
        while low <= high:
            mid = (low + high) // 2
            if global_index < self.cumulative_lengths[mid]:
                high = mid - 1
            else:
                low = mid + 1
        return low


def collate_draft_gt(batch):
    drafts = [b["draft"] for b in batch]
    gts = [b["gt"] for b in batch]
    return {
        "draft": torch.cat(drafts, dim=0),
        "gt": torch.cat(gts, dim=0)
    }


###############################################################################
# 推理（采样）函数
###############################################################################
def inference_pipeline(draft_hidden, noise_scheduler, model, device, num_inference_steps=None):
    if num_inference_steps is None:
        num_inference_steps = noise_scheduler.config.num_train_timesteps

    batch_size = draft_hidden.shape[0]
    x_t = torch.randn_like(draft_hidden, device=device)

    for t in reversed(range(num_inference_steps)):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        output = model(sample=x_t, timestep=t_tensor, draft=draft_hidden, return_dict=True)
        pred_x0 = output.sample
        step_output = noise_scheduler.step(pred_x0, t, x_t)
        x_t = step_output.prev_sample

    return x_t


###############################################################################
# 列出某个文件夹下的所有文件
###############################################################################
def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath


###############################################################################
# 评估草稿 (draft) 的初始 top-k 准确率
###############################################################################
def evaluate_draft_before_training(dataloader, head, accelerator):
    logger.info("Start evaluating draft before training ...")
    device = accelerator.device
    head.eval()

    top1_in_top1_sum = 0.0
    top1_in_top2_sum = 0.0
    top1_in_top3_sum = 0.0

    total_samples = 0
    total_mse = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluate Draft", ncols=80):
            draft = batch["draft"].to(device)
            gt    = batch["gt"].to(device)
            bsz   = draft.size(0)

            # draft -> LM Head
            pred_logits = head(draft)
            # gt -> LM Head
            gt_logits = head(gt)

            # draft vs gt 的 MSE
            mse = F.mse_loss(draft, gt).item()
            total_mse += mse * bsz

            # 计算 top-k
            pred_top3_tokens = torch.topk(pred_logits, k=3, dim=-1).indices
            gt_top3_tokens   = torch.topk(gt_logits, k=3, dim=-1).indices

            pred_top1_tokens = pred_top3_tokens[:, 0]
            gt_top1_tokens   = gt_top3_tokens[:, 0]
            gt_top2_tokens   = gt_top3_tokens[:, :2]

            for i in range(bsz):
                if pred_top1_tokens[i] == gt_top1_tokens[i]:
                    top1_in_top1_sum += 1
                if pred_top1_tokens[i] in gt_top2_tokens[i]:
                    top1_in_top2_sum += 1
                if pred_top1_tokens[i] in gt_top3_tokens[i]:
                    top1_in_top3_sum += 1

            total_samples += bsz

    if total_samples == 0:
        avg_mse = 0.0
        top1_in_top1 = 0.0
        top1_in_top2 = 0.0
        top1_in_top3 = 0.0
    else:
        avg_mse = total_mse / total_samples
        top1_in_top1 = top1_in_top1_sum / total_samples
        top1_in_top2 = top1_in_top2_sum / total_samples
        top1_in_top3 = top1_in_top3_sum / total_samples

    logger.info(f"[Draft] MSE(draft, gt): {avg_mse:.4f}")
    logger.info(f"[Draft] Top1 in Top1 Accuracy: {top1_in_top1:.4f}")
    logger.info(f"[Draft] Top1 in Top2 Accuracy: {top1_in_top2:.4f}")
    logger.info(f"[Draft] Top1 in Top3 Accuracy: {top1_in_top3:.4f}")

    if accelerator.is_main_process:
        wandb.log({
            "val_loss": avg_mse,
            "top1_in_top1_accuracy": top1_in_top1,
            "top1_in_top2_accuracy": top1_in_top2,
            "top1_in_top3_accuracy": top1_in_top3
        }, step=0)



###############################################################################
# 解析参数
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Train and Inference for Conditional Hidden-State Diffusion")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="存放包含 draft 和 gt 的 hidden state 文件的文件夹")
    parser.add_argument("--train_ratio", type=float, default=0.998, help="训练集占比")
    parser.add_argument("--output_dir", type=str, default="ckpts")
    parser.add_argument("--train_batch_size", type=int, default=8192)
    parser.add_argument("--hidden_dim", type=int, default=4096, help="draft 和 gt 的隐藏向量维度")
    parser.add_argument("--time_embed_dim", type=int, default=64, help="时间步嵌入的维度")
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    ##### 新增参数：warmup_steps（可改成 warmup_ratio 视需要）
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--basepath", type=str, default=None, 
                        help="预训练 LM Head 的路径（可选），若不指定则跳过 LM Head 的加载")
    return parser.parse_args()


###############################################################################
# 主函数
###############################################################################
def main():
    args = parse_args()

    from datetime import datetime
    run_name = datetime.now().strftime("%Y%m%d_%H%M")
    project_dir = os.path.join(args.output_dir, run_name)

    # 初始化 Accelerate
    accelerator_project_config = ProjectConfiguration(project_dir=project_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    # 在主进程中初始化 wandb
    if accelerator.is_main_process:
        wandb.init(
            project="SpecDiff",
            entity="reflectionie",
            name=run_name,
            config=vars(args)
        )

    # 仅需列出 data_dir 下的所有文件
    all_paths = sorted(list_files(args.data_dir))
    total_files = len(all_paths)
    train_size = int(total_files * args.train_ratio)

    train_paths = all_paths[:train_size]
    val_paths = all_paths[train_size:]
    logger.info(f"Total files: {total_files}, Train files: {train_size}, Val files: {total_files - train_size}")

    # 构建 dataset 与 dataloader
    train_dataset = ChainedHiddenStateBatchDataset(train_paths, args.train_batch_size)
    val_dataset = ChainedHiddenStateBatchDataset(val_paths, args.train_batch_size)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_draft_gt)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_draft_gt)

    # 初始化条件扩散模型
    model = HiddenStateDiffusionModel(hidden_dim=args.hidden_dim, time_embed_dim=args.time_embed_dim)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total training parameters: {total_params/1e9:.4f}B")

    model.train()

    # 调度器
    num_train_timesteps = 1000
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=1e-4,
        beta_end=1.9e-3,
        beta_schedule="linear",
        prediction_type="sample"
    )

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 先不定义 scheduler，先准备好基础组件 ==========
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    # 计算总训练步数
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    ##### 新增：定义学习率调度器（以 linear warmup+decay 为例）
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,       # 线性预热 steps（可按需求改）
        num_training_steps=args.max_train_steps   # 训练总步数
    )

    ##### 新增：将 scheduler 也交给 Accelerator 来 prepare
    scheduler = accelerator.prepare(scheduler)

    if accelerator.is_main_process:
        os.makedirs(project_dir, exist_ok=True)

    # ========== 加载 LM Head（可选） =========
    head = None
    if args.basepath is not None:
        logger.info(f"Loading LM Head from: {args.basepath}")
        baseconfig = AutoConfig.from_pretrained(args.basepath)
        head = torch.nn.Linear(baseconfig.hidden_size, baseconfig.vocab_size, bias=False)

        try:
            # safetensors 加载
            with open(os.path.join(args.basepath, "model.safetensors.index.json"), "r") as f:
                index_json = json.loads(f.read())
                head_path = index_json["weight_map"]["lm_head.weight"]
            with safe_open(os.path.join(args.basepath, head_path), framework="pt", device="cpu") as f:
                tensor_slice = f.get_slice("lm_head.weight")
                vocab_size, hidden_dim = tensor_slice.get_shape()
                tensor = tensor_slice[:, :hidden_dim].float()
        except:
            # pytorch_model.bin 加载
            with open(os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r") as f:
                index_json = json.loads(f.read())
                head_path = index_json["weight_map"]["lm_head.weight"]
            weights = torch.load(os.path.join(args.basepath, head_path))
            tensor = weights["lm_head.weight"].float()

        head.weight.data = tensor
        head.eval()
        for param in head.parameters():
            param.requires_grad = False
        head = accelerator.prepare(head)
        logger.info("LM Head loaded and prepared by Accelerate.")
    else:
        logger.info("LM Head evaluation disabled (no basepath provided).")

    # ========== 在开始训练之前，先对 draft 做一次评估 ========== 
    if head is not None:
        evaluate_draft_before_training(val_dataloader, head, accelerator)
    else:
        logger.info("Skip draft evaluation because head is None.")

    # ========== 正式训练 ========== 
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, ncols=80)
    progress_bar.set_description("Training")

    global_step = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            x_draft = batch["draft"]
            x_gt = batch["gt"]
            bsz = x_gt.size(0)

            with accelerator.accumulate(model):
                # 1) 随机时间步 + 加噪
                timesteps = torch.randint(0, num_train_timesteps, (bsz,), device=x_gt.device)
                noise = torch.randn_like(x_gt)
                x_noisy = noise_scheduler.add_noise(x_gt, noise, timesteps)

                # 2) 前向 & 损失
                output = model(sample=x_noisy, timestep=timesteps, draft=x_draft, return_dict=True)
                predicted_gt = output.sample
                loss = F.mse_loss(predicted_gt, x_gt.to(predicted_gt.dtype))

                # 3) 反向 & 优化
                accelerator.backward(loss)
                optimizer.step()
                ##### 新增：每次同步梯度后再对 scheduler 进行 step
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad()

            # 只有在同步梯度时才更新进度与 global_step
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    wandb.log({"train_loss": loss.item(), "global_step": global_step})

                # ========== 周期性验证与保存 ========== 
                if global_step % args.checkpointing_steps == 0:
                    model.eval()
                    val_loss = 0.0
                    val_samples = 0

                    top1_in_top1_sum = 0.0
                    top1_in_top2_sum = 0.0
                    top1_in_top3_sum = 0.0

                    with torch.no_grad():
                        for val_batch in tqdm(val_dataloader, desc="Validating", ncols=80):
                            x_draft_val = val_batch["draft"]
                            x_gt_val = val_batch["gt"]
                            bsz_val = x_draft_val.size(0)

                            pred_gt = inference_pipeline(
                                draft_hidden=x_draft_val,
                                noise_scheduler=noise_scheduler,
                                model=model,
                                device=x_draft_val.device,
                                num_inference_steps=num_train_timesteps,
                            )

                            loss_val = F.mse_loss(pred_gt, x_gt_val)
                            val_loss += loss_val.item() * bsz_val
                            val_samples += bsz_val

                            if head is not None:
                                pred_logits = head(pred_gt)
                                gt_logits   = head(x_gt_val)

                                pred_top3_tokens = torch.topk(pred_logits, k=3, dim=-1).indices
                                gt_top3_tokens   = torch.topk(gt_logits, k=3, dim=-1).indices

                                pred_top1_tokens = pred_top3_tokens[:, 0]
                                gt_top1_tokens   = gt_top3_tokens[:, 0]
                                gt_top2_tokens   = gt_top3_tokens[:, :2]

                                for i in range(bsz_val):
                                    if pred_top1_tokens[i] == gt_top1_tokens[i]:
                                        top1_in_top1_sum += 1
                                    if pred_top1_tokens[i] in gt_top2_tokens[i]:
                                        top1_in_top2_sum += 1
                                    if pred_top1_tokens[i] in gt_top3_tokens[i]:
                                        top1_in_top3_sum += 1

                    # 计算验证集上总的指标
                    if val_samples > 0:
                        avg_val_loss = val_loss / val_samples
                        top1_in_top1_prob = top1_in_top1_sum / val_samples
                        top1_in_top2_prob = top1_in_top2_sum / val_samples
                        top1_in_top3_prob = top1_in_top3_sum / val_samples
                    else:
                        avg_val_loss = 0.0
                        top1_in_top1_prob = 0.0
                        top1_in_top2_prob = 0.0
                        top1_in_top3_prob = 0.0

                    logger.info(f"[Global Step {global_step}] Val Loss: {avg_val_loss:.4f}")
                    logger.info(f"[Global Step {global_step}] Top1 in Top1: {top1_in_top1_prob:.4f}")
                    logger.info(f"[Global Step {global_step}] Top1 in Top2: {top1_in_top2_prob:.4f}")
                    logger.info(f"[Global Step {global_step}] Top1 in Top3: {top1_in_top3_prob:.4f}")

                    if accelerator.is_main_process:
                        wandb.log({
                            "val_loss": avg_val_loss,
                            "top1_in_top1_accuracy": top1_in_top1_prob,
                            "top1_in_top2_accuracy": top1_in_top2_prob,
                            "top1_in_top3_accuracy": top1_in_top3_prob,
                            "global_step": global_step,
                            ##### 记录学习率
                            "learning_rate": scheduler.get_last_lr()[0]
                        })

                        # 保存checkpoint
                        # ckpt_path = os.path.join(project_dir, f"pytorch_model_step{global_step}.bin")
                        # unwrapped_model = accelerator.unwrap_model(model)
                        # torch.save(unwrapped_model.state_dict(), ckpt_path)
                        # logger.info(f"Checkpoint saved at step {global_step} -> {ckpt_path}")

                    model.train()

            # 提前退出
            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    logger.info("Training loop complete.")

    # 可选：保存最终模型
    if accelerator.is_main_process:
        final_path = os.path.join(project_dir, "pytorch_model_final.bin")
        torch.save(accelerator.unwrap_model(model).state_dict(), final_path)
        logger.info(f"Final model saved to {final_path}")

    logger.info("Training & Inference finished.")


if __name__ == "__main__":
    main()
