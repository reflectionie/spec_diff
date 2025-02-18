#!/usr/bin/env python
# coding=utf-8
"""
python train_hidden_state_diffuser.py \
  --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct \
  --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data \
  --checkpointing_steps 1000

"""

import argparse
import logging
import math
import os
from pathlib import Path
import json
from safetensors import safe_open
from transformers import AutoConfig

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
from my_hidden_state_diffusion import HiddenStateDiffusionModel, HiddenStateDiffusionOutput

logger = get_logger(__name__, log_level="INFO")


###############################################################################
# 新版数据集：同一个文件里同时包含 "draft_hidden" 和 "hidden_state"
###############################################################################
class ChainedHiddenStateBatchDataset(Dataset):
    """
    该数据集将多个文件中的 hidden state 按顺序拼接成一个长序列，
    然后每次返回一个连续片段，该片段的长度等于 batch_size（即 chunk_size）。

    假设每个文件的内容格式为：
        {
           "draft_hidden": tensor of shape [seq_len, hidden_dim],
           "hidden_state": tensor of shape [seq_len, hidden_dim]
        }
    那么我们会将所有文件的草稿 hidden_state (draft) 和真实 hidden_state (gt)
    各自拼接成一个按顺序的长序列。
    """

    def __init__(self, paths, chunk_size, num_workers=10):
        """
        :param paths: 文件路径列表，每个文件包含 {"draft_hidden", "hidden_state"} 两个键
        :param chunk_size: 训练时每个 batch 需要的连续长度
        :param num_workers: 多线程加载文件时使用的线程数
        """
        super().__init__()
        self.paths = paths
        self.chunk_size = chunk_size

        # 预先计算每个文件中的 hidden state 数量，以及累积长度
        self.file_lengths = []
        self.cumulative_lengths = []
        cumulative = 0

        print("Counting data total_length with multi-threading...")

        def get_file_length(p):
            # 只加载CPU版本，减少内存占用
            data = torch.load(p, map_location="cpu")
            # 假设 draft_hidden 与 hidden_state 的长度相同
            return data["draft_hidden"].shape[0]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            file_lengths = list(tqdm(executor.map(get_file_length, self.paths), total=len(self.paths)))
        
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
        """
        返回一个训练 batch，包含连续的 chunk_size 个 hidden state（分别来自 draft 和 gt）。
        形状:
            draft: [chunk_size, hidden_dim]
            gt: [chunk_size, hidden_dim]
        """
        # 全局序列的起始和结束位置
        start = index * self.chunk_size
        end = start + self.chunk_size

        draft_chunks = []
        gt_chunks = []
        current_global = start

        while current_global < end:
            file_idx = self._find_file_index(current_global)
            file_start_global = self.cumulative_lengths[file_idx - 1] if file_idx > 0 else 0
            local_start = current_global - file_start_global

            # 当前文件可用的数量
            available = self.file_lengths[file_idx] - local_start
            needed = end - current_global
            take = min(available, needed)

            # 只需加载一次文件
            loaded = torch.load(self.paths[file_idx])
            draft_data = loaded["draft_hidden"]
            gt_data = loaded["hidden_state"]

            draft_chunks.append(draft_data[local_start: local_start + take])
            gt_chunks.append(gt_data[local_start: local_start + take])

            current_global += take

        draft_batch = torch.cat(draft_chunks, dim=0)
        gt_batch = torch.cat(gt_chunks, dim=0)

        # 确保拼接后的数量正好为 chunk_size
        assert draft_batch.shape[0] == self.chunk_size
        assert gt_batch.shape[0] == self.chunk_size

        return {"draft": draft_batch, "gt": gt_batch}

    def _find_file_index(self, global_index):
        """
        二分查找：根据全局索引找到该 hidden state 属于哪个文件
        返回文件的索引
        """
        low, high = 0, len(self.cumulative_lengths) - 1
        while low <= high:
            mid = (low + high) // 2
            if global_index < self.cumulative_lengths[mid]:
                high = mid - 1
            else:
                low = mid + 1
        return low


def collate_draft_gt(batch):
    """
    将一个batch (List[{"draft": tensor, "gt": tensor}]) 拼接。
    最终返回:
    {
        "draft": [B * chunk_size, hidden_dim],
        "gt": [B * chunk_size, hidden_dim]
    }
    注意: 这里的 B 通常是 1（因为我们的 Dataset 直接按 chunk_size 返回数据），
         但也可以适配更一般的场景。
    """
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
    """
    给定 draft_hidden 作为条件，从纯噪声开始，通过反向扩散生成预测的 gt_hidden。
    """
    if num_inference_steps is None:
        num_inference_steps = noise_scheduler.config.num_train_timesteps

    batch_size = draft_hidden.shape[0]
    x_t = torch.randn_like(draft_hidden, device=device)  # x_T

    for t in tqdm(reversed(range(num_inference_steps))):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        output = model(sample=x_t, timestep=t_tensor, draft=draft_hidden, return_dict=True)
        pred_x0 = output.sample
        step_output = noise_scheduler.step(pred_x0, t, x_t)
        x_t = step_output.prev_sample

    return x_t


###############################################################################
# 工具函数：列出某个文件夹下的所有文件
###############################################################################
def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath


###############################################################################
# 解析参数
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Train and Inference for Conditional Hidden-State Diffusion")
    # 仅保留一个 data_dir 用于存放同时含 draft_hidden 和 hidden_state 的文件
    parser.add_argument("--data_dir", type=str, required=True, help="存放包含 draft 和 gt 的 hidden state 文件的文件夹")
    parser.add_argument("--train_ratio", type=float, default=0.95, help="训练集占比")
    parser.add_argument("--output_dir", type=str, default="ckpts")
    parser.add_argument("--train_batch_size", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=4096, help="draft 和 gt 的隐藏向量维度")
    parser.add_argument("--time_embed_dim", type=int, default=64, help="时间步嵌入的维度")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
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

    # 初始化 Accelerate
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
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
        wandb.init(project="SpecDiff", entity="reflectionie", config=vars(args))

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

    # 参数数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total training parameters: {total_params/1e9:.4f}B")

    model.train()

    # 调度器
    num_train_timesteps = 1000
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="sample"
    )

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Accelerate 包装
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    # 计算总训练步数
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training")

    # ========= 加载 LM Head（可选） =========
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
        head = None
        logger.info("LM Head evaluation disabled (no basepath provided).")

    # ========== 训练 ==========
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training")

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            x_draft = batch["draft"]
            x_gt = batch["gt"]
            bsz = x_gt.size(0)

            with accelerator.accumulate(model):
                # 1. 取时间步、加噪
                timesteps = torch.randint(0, num_train_timesteps, (bsz,), device=x_gt.device)
                noise = torch.randn_like(x_gt)
                x_noisy = noise_scheduler.add_noise(x_gt, noise, timesteps)

                # 2. 前向与损失
                output = model(
                    sample=x_noisy,
                    timestep=timesteps,
                    draft=x_draft,
                    return_dict=True
                )
                predicted_gt = output.sample
                loss = F.mse_loss(predicted_gt, x_gt.to(predicted_gt.dtype))

                # 3. 反向与优化
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            # 只有在同步梯度时才更新进度与 global_step
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # 记录训练loss
                if accelerator.is_main_process:
                    wandb.log({"train_loss": loss.item(), "global_step": global_step})

                # ========== 【修改点】按checkpointing_steps进行验证和保存 ==========
                if global_step % args.checkpointing_steps == 0:
                    # 验证
                    model.eval()
                    val_loss = 0.0
                    val_samples = 0

                    top1_in_top1_sum = 0.0
                    top1_in_top2_sum = 0.0
                    top1_in_top3_sum = 0.0

                    with torch.no_grad():
                        for val_batch in val_dataloader:
                            x_draft_val = val_batch["draft"]
                            x_gt_val = val_batch["gt"]
                            bsz_val = x_draft_val.size(0)

                            # 推理
                            pred_gt = inference_pipeline(
                                draft_hidden=x_draft_val,
                                noise_scheduler=noise_scheduler,
                                model=model,
                                device=x_draft_val.device,
                                num_inference_steps=num_train_timesteps,
                            )

                            # 验证loss
                            loss_val = F.mse_loss(pred_gt, x_gt_val)
                            val_loss += loss_val.item() * bsz_val
                            val_samples += bsz_val

                            # 若有 LM Head，则做 top-k 的简单评估
                            if head is not None:
                                pred_logits = head(pred_gt)
                                gt_logits = head(x_gt_val)

                                pred_top3_tokens = torch.topk(pred_logits, k=3, dim=-1).indices
                                gt_top3_tokens = torch.topk(gt_logits, k=3, dim=-1).indices
                                pred_top1_tokens = pred_top3_tokens[:, 0]
                                gt_top1_tokens = gt_top3_tokens[:, 0]
                                gt_top2_tokens = gt_top3_tokens[:, :2]

                                for i in range(bsz_val):
                                    if pred_top1_tokens[i] == gt_top1_tokens[i]:
                                        top1_in_top1_sum += 1
                                    if pred_top1_tokens[i] in gt_top2_tokens[i]:
                                        top1_in_top2_sum += 1
                                    if pred_top1_tokens[i] in gt_top3_tokens[i]:
                                        top1_in_top3_sum += 1

                    # 计算验证指标
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
                    logger.info(f"[Global Step {global_step}] Top-1 in Top-1: {top1_in_top1_prob:.4f}")
                    logger.info(f"[Global Step {global_step}] Top-1 in Top-2: {top1_in_top2_prob:.4f}")
                    logger.info(f"[Global Step {global_step}] Top-1 in Top-3: {top1_in_top3_prob:.4f}")

                    if accelerator.is_main_process:
                        wandb.log({
                            "val_loss": avg_val_loss,
                            "top1_in_top1_accuracy": top1_in_top1_prob,
                            "top1_in_top2_accuracy": top1_in_top2_prob,
                            "top1_in_top3_accuracy": top1_in_top3_prob,
                            "global_step": global_step
                        })

                    # 保存当前checkpoint
                    # 仅主进程保存模型；可根据需求保存在不同目录
                    if accelerator.is_main_process:
                        ckpt_path = os.path.join(args.output_dir, f"pytorch_model_step{global_step}.bin")
                        unwrapped_model = accelerator.unwrap_model(model)
                        torch.save(unwrapped_model.state_dict(), ckpt_path)
                        logger.info(f"Checkpoint saved at step {global_step} -> {ckpt_path}")

                    model.train()  # 验证后切回 train 模式

            # 如果达到max_train_steps，就停止
            if global_step >= args.max_train_steps:
                break

        logger.info("Training loop complete.")

        # 训练结束后，可选地再做一次验证或保存最终模型（根据需要）
        if accelerator.is_main_process:
            final_path = os.path.join(args.output_dir, "pytorch_model_final.bin")
            torch.save(accelerator.unwrap_model(model).state_dict(), final_path)
            logger.info(f"Final model saved to {final_path}")


    logger.info("Training & Inference finished.")


if __name__ == "__main__":
    main()
