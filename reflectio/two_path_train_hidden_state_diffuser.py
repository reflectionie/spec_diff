#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import math
import os
from pathlib import Path
import json  # 导入 json 库
from safetensors import safe_open # 导入 safe_open
from transformers import AutoConfig # 导入 AutoConfig

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm

import wandb  # 新增 wandb 导入

from diffusers import DDPMScheduler

# 导入我们定义的条件扩散模型，要求 draft 和 gt 的 hidden_dim 保持一致
from my_hidden_state_diffusion import HiddenStateDiffusionModel, HiddenStateDiffusionOutput

logger = get_logger(__name__, log_level="INFO")

import torch
from torch.utils.data import Dataset

class ChainedHiddenStateBatchDataset(Dataset):
    """
    该数据集将多个文件中的 hidden state 按顺序拼接成一个长序列，
    然后每次返回一个连续片段，该片段的长度等于 batch_size（即 chunk_size）。

    假设每个文件的内容格式为 { "hidden_state": tensor }，tensor 的形状为 [seq_len, hidden_dim]，
    那么整个数据集拼接后的形状是 [total_hidden_states, hidden_dim]，
    每次 __getitem__ 返回一个形状为 [chunk_size, hidden_dim] 的片段。

    这样在训练时，可以直接使用 DataLoader(batch_size=1) 或者不使用 DataLoader 的 batch 功能，
    而将每个返回的片段视作一个完整的训练 batch，从而避免单个 batch 的样本分别来自多个文件，频繁加载多个文件。
    """
    def __init__(self, draft_paths, gt_paths, chunk_size):
        """
        :param draft_paths: 存放 draft hidden state 文件的路径列表
        :param gt_paths: 存放 gt hidden state 文件的路径列表（与 draft_paths 一一对应）
        :param chunk_size: 每个 batch 需要连续取出的 hidden state 数量，即训练时的 batch_size
        """
        super().__init__()
        assert len(draft_paths) == len(gt_paths), "draft/gt 文件数量不匹配"
        self.draft_paths = draft_paths
        self.gt_paths = gt_paths
        self.chunk_size = chunk_size

        # 预先计算每个文件中的 hidden state 数量，以及全局位置（累积长度）
        self.file_lengths = []      # 每个文件的 hidden state 数量
        self.cumulative_lengths = []  # 累计长度列表，用于快速定位全局索引对应哪个文件
        cumulative = 0
        for d_path, g_path in zip(self.draft_paths, self.gt_paths):
            # 这里只加载一次，假设 draft 和 gt 的长度相同
            data = torch.load(d_path)
            length = data["hidden_state"].shape[0]
            self.file_lengths.append(length)
            cumulative += length
            self.cumulative_lengths.append(cumulative)
        self.total_length = cumulative

        # 按照 chunk_size 划分后，总共可以构成多少个完整的 batch
        self.num_batches = self.total_length // self.chunk_size

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        """
        返回一个训练 batch，包含连续的 chunk_size 个 hidden state（来自 draft 和 gt）
        输出：
            {
                "draft": tensor of shape [chunk_size, hidden_dim],
                "gt": tensor of shape [chunk_size, hidden_dim]
            }
        """
        # 全局序列的起始和结束位置
        start = index * self.chunk_size
        end = start + self.chunk_size

        draft_chunks = []
        gt_chunks = []
        current_global = start

        # 从全局序列中截取[start, end)范围内的 hidden state
        # 可能需要跨越多个文件来构造一个完整的 batch
        while current_global < end:
            file_idx = self._find_file_index(current_global)
            # 当前文件在全局序列中的起始位置
            file_start_global = self.cumulative_lengths[file_idx - 1] if file_idx > 0 else 0
            # 当前文件内的起始索引
            local_start = current_global - file_start_global

            # 当前文件中还剩余多少个 hidden state
            available = self.file_lengths[file_idx] - local_start
            needed = end - current_global
            take = min(available, needed)

            # 加载当前文件数据
            draft_data = torch.load(self.draft_paths[file_idx])["hidden_state"]
            gt_data = torch.load(self.gt_paths[file_idx])["hidden_state"]

            draft_chunks.append(draft_data[local_start: local_start + take])
            gt_chunks.append(gt_data[local_start: local_start + take])

            current_global += take

        # 拼接来自不同文件的片段，得到完整的 batch
        draft_batch = torch.cat(draft_chunks, dim=0)
        gt_batch = torch.cat(gt_chunks, dim=0)

        # 安全检查：确保拼接后的数量正好为 chunk_size
        assert draft_batch.shape[0] == self.chunk_size, (
            f"draft_batch 的长度为 {draft_batch.shape[0]}，期望 {self.chunk_size}"
        )
        assert gt_batch.shape[0] == self.chunk_size, (
            f"gt_batch 的长度为 {gt_batch.shape[0]}，期望 {self.chunk_size}"
        )

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



###############################################################################
# 数据集定义：LazyDraftGtFlattenedDataset
###############################################################################
class LazyDraftGtFlattenedDataset(Dataset):
    """
    懒加载数据集：仅存储 draft 和 gt 文件的路径以及每个文件中 hidden_state 的行数，
    在 __getitem__ 时根据全局索引计算出对应文件及行索引，加载后返回该行数据。

    每个文件内部的 hidden_state（形状 [seq_len, hidden_dim]）会被逐行拆分，
    返回的样本为一个单独的 hidden state 向量，形状 [hidden_dim]。
    """
    def __init__(self, draft_paths, gt_paths, transform=None):
        super().__init__()
        assert len(draft_paths) == len(gt_paths), "draft/gt 文件列表长度不匹配"
        self.draft_paths = draft_paths
        self.gt_paths = gt_paths
        self.transform = transform

        self.cumulative_lengths = []
        cumulative = 0
        for d_path, g_path in zip(self.draft_paths, self.gt_paths):
            draft_data = torch.load(d_path)
            gt_data = torch.load(g_path)
            length = min(draft_data["hidden_state"].shape[0],
                         gt_data["hidden_state"].shape[0])
            cumulative += length
            self.cumulative_lengths.append(cumulative)
        self.total_samples = cumulative

    def __len__(self):
        return self.total_samples

    def _get_file_index(self, idx):
        # 二分查找：找到满足 cumulative_lengths[i-1] <= idx < cumulative_lengths[i] 的 i
        low, high = 0, len(self.cumulative_lengths) - 1
        while low <= high:
            mid = (low + high) // 2
            if idx < self.cumulative_lengths[mid]:
                high = mid - 1
            else:
                low = mid + 1
        return low

    def __getitem__(self, idx):
        file_index = self._get_file_index(idx)
        row_index = idx if file_index == 0 else idx - self.cumulative_lengths[file_index - 1]

        draft_data = torch.load(self.draft_paths[file_index])["hidden_state"]
        gt_data = torch.load(self.gt_paths[file_index])["hidden_state"]
        min_len = min(draft_data.shape[0], gt_data.shape[0])
        if row_index >= min_len:
            row_index = min_len - 1

        sample = {"draft": draft_data[row_index], "gt": gt_data[row_index]}
        if self.transform:
            sample = self.transform(sample)
        return sample


def collate_draft_gt(batch):
    """
    将一个 batch（List[{"draft": tensor, "gt": tensor}])
    整理为：
      {"draft": [B, hidden_dim], "gt": [B, hidden_dim]}
    注意这里用 torch.cat 而不是 torch.stack，这样如果每个 item 已经是一个完整 batch（例如 [2048, hidden_dim]），
    最终返回的就不会额外增加一个维度。
    """
    drafts = [b["draft"] for b in batch]
    gts = [b["gt"] for b in batch]
    return {"draft": torch.cat(drafts, dim=0), "gt": torch.cat(gts, dim=0)}



###############################################################################
# 推理（采样）函数
###############################################################################
def inference_pipeline(draft_hidden, noise_scheduler, model, device, num_inference_steps=None):
    """
    给定 draft_hidden (形状 [B, hidden_dim]) 作为条件，
    从纯噪声开始，通过反向扩散生成预测的 gt_hidden。

    参数：
      draft_hidden: [B, hidden_dim]，条件不变
      noise_scheduler: DDPMScheduler
      model: 条件扩散模型（输入 sample, timestep, draft）
      device: torch.device
      num_inference_steps: 推理时使用的扩散步数

    返回：
      x_0: 预测的 gt_hidden, 形状 [B, hidden_dim]
    """
    if num_inference_steps is None:
        num_inference_steps = noise_scheduler.config.num_train_timesteps

    batch_size = draft_hidden.shape[0]
    # 从标准高斯分布采样 x_T
    x_t = torch.randn_like(draft_hidden, device=device)

    for t in tqdm(reversed(range(num_inference_steps))):
        # t 现在是一个标量整数时间步
        # **修改点：生成批量大小的时间步 Tensor**
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # 模型预测噪声，注意此处传入条件 draft_hidden
        output = model(sample=x_t, timestep=t_tensor, draft=draft_hidden, return_dict=True)
        pred_x0  = output.sample  # 形状 [B, hidden_dim]

        # 用调度器一步反扩散
        step_output = noise_scheduler.step(pred_x0, t, x_t) #  直接传递标量 t 给 step 函数
        x_t = step_output.prev_sample

    return x_t  # 预测的 x_0，期望逼近 gt_hidden

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
    parser.add_argument("--draft_dir", type=str, required=True, help="存放 draft hidden state 文件的文件夹")
    parser.add_argument("--gt_dir", type=str, required=True, help="存放 gt hidden state 文件的文件夹")
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
    parser.add_argument("--basepath", type=str, default="/net/graphium/storage3/tingyuan/models/Meta-Llama-3-8B-Instruct", help="预训练 LM Head 的路径") # 新增 basepath 参数
    # 移除了 report_to 参数，不再使用 tensorboard
    return parser.parse_args()


###############################################################################
# 主函数：训练 & 推理
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
        wandb.init(
        project="SpecDiff",
        entity="reflectionie",
        config=vars(args)
        )

    # 准备文件列表并拆分（按照文件数划分）
    draft_paths_all = sorted(list_files(args.draft_dir))
    gt_paths_all = sorted(list_files(args.gt_dir))

    total_files = len(draft_paths_all)
    train_size = int(total_files * args.train_ratio)
    train_draft_paths = draft_paths_all[:train_size]
    train_gt_paths = gt_paths_all[:train_size]
    val_draft_paths = draft_paths_all[train_size:]
    val_gt_paths = gt_paths_all[train_size:]

    logger.info(f"Total files: {total_files}, Train files: {train_size}, Val files: {total_files - train_size}")

    # 构建 dataset 与 dataloader
    train_dataset = ChainedHiddenStateBatchDataset(train_draft_paths, train_gt_paths, args.train_batch_size)
    val_dataset = ChainedHiddenStateBatchDataset(val_draft_paths, val_gt_paths, args.train_batch_size)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_draft_gt)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_draft_gt)

    # 初始化条件扩散模型（draft 与 gt 维度均为 hidden_dim，不改变维度）
    model = HiddenStateDiffusionModel(hidden_dim=args.hidden_dim, time_embed_dim=args.time_embed_dim)


    # 计算并打印模型的可训练参数量，单位为B（billion）
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total training parameters: {total_params/1e9:.4f}B")

    model.train()


    # 调度器：DDPMScheduler
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

    # 使用 Accelerate 包装模型、优化器与 dataloader
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader)

    # 计算总训练步数
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training")

    # ========== 加载 LM Head ==========
    if args.basepath is not None: # 只有当 basepath 参数被指定时才加载 LM Head
        logger.info(f"Loading LM Head from: {args.basepath}")
        baseconfig = AutoConfig.from_pretrained(args.basepath)
        head = torch.nn.Linear(baseconfig.hidden_size, baseconfig.vocab_size, bias=False)

        try:
            with open(os.path.join(args.basepath, "model.safetensors.index.json"), "r") as f:
                index_json = json.loads(f.read())
                head_path = index_json["weight_map"]["lm_head.weight"]
            with safe_open(os.path.join(args.basepath, head_path),
                            framework="pt",
                            device="cpu") as f:
                tensor_slice = f.get_slice("lm_head.weight")
                vocab_size, hidden_dim = tensor_slice.get_shape()
                tensor = tensor_slice[:, :hidden_dim].float() # 确保切片到 hidden_dim 大小
        except:
            with open(os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r") as f:
                index_json = json.loads(f.read())
                head_path = index_json["weight_map"]["lm_head.weight"]
            weights = torch.load(os.path.join(args.basepath, head_path))
            tensor = weights["lm_head.weight"].float()

        head.weight.data = tensor
        head.eval()
        for param in head.parameters():
            param.requires_grad = False
        head = accelerator.prepare(head) # 使用accelerator prepare
        logger.info("LM Head loaded and prepared by Accelerate.")
    else:
        head = None # 如果没有指定 basepath，则 head 为 None
        logger.info("LM Head evaluation disabled as no basepath provided.")


    # ========== 训练 ==========
    for epoch in tqdm(range(args.num_train_epochs)):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # 从 dataset 中取出 draft 和 gt，形状均为 [B, hidden_dim]
            x_draft = batch["draft"]
            x_gt = batch["gt"]

            with accelerator.accumulate(model):
                bsz = x_gt.size(0)
                # 随机采样时间步 t (形状 [B])
                timesteps = torch.randint(0, num_train_timesteps, (bsz,), device=x_gt.device)
                # 加噪：依然从 gt 加噪得到 x_noisy
                noise = torch.randn_like(x_gt)
                x_noisy = noise_scheduler.add_noise(x_gt, noise, timesteps)

                # 模型输入：sample=x_noisy, timestep, draft=x_draft
                output = model(sample=x_noisy, timestep=timesteps, draft=x_draft, return_dict=True)
                predicted_gt = output.sample  # 网络输出直接是预测的 gt

                # 损失：比较预测的 gt 与真实 gt
                loss = F.mse_loss(predicted_gt, x_gt.to(predicted_gt.dtype))

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # 仅在主进程中记录训练 loss 到 wandb
                if accelerator.is_main_process:
                    wandb.log({"train_loss": loss.item(), "global_step": global_step})

            if global_step >= args.max_train_steps:
                break

        logger.info(f"Epoch {epoch + 1} finished.")

        # ========== 验证 / 推理 ==========
        model.eval()
        val_loss = 0.0
        val_samples = 0
        top1_in_top1_sum = 0.0 # 预测 top1 在 gt top1 中的次数
        top1_in_top2_sum = 0.0 # 预测 top1 在 gt top2 中的次数
        top1_in_top3_sum = 0.0 # 预测 top1 在 gt top3 中的次数


        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                x_draft_val = batch["draft"]
                x_gt_val = batch["gt"]
                bsz_val = x_draft_val.size(0)

                # 使用推理流程：给定 x_draft_val 作为条件，从噪声开始逐步还原
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


                if head is not None: # 如果加载了 LM Head，则进行评估
                    # 使用 LM Head 解码 hidden state
                    pred_logits = head(pred_gt) # [B, vocab_size]
                    gt_logits = head(x_gt_val)     # [B, vocab_size]

                    # 获取 top-k token
                    pred_top3_tokens = torch.topk(pred_logits, k=3, dim=-1).indices # [B, 3]
                    gt_top3_tokens = torch.topk(gt_logits, k=3, dim=-1).indices     # [B, 3]
                    pred_top1_tokens = pred_top3_tokens[:, 0] # [B]
                    gt_top1_tokens = gt_top3_tokens[:, 0]     # [B]
                    gt_top2_tokens = gt_top3_tokens[:, :2]     # [B, 2]


                    # 计算 top-k 准确率
                    for i in range(bsz_val):
                        if pred_top1_tokens[i] == gt_top1_tokens[i]:
                            top1_in_top1_sum += 1
                        if pred_top1_tokens[i] in gt_top2_tokens[i]:
                            top1_in_top2_sum += 1
                        if pred_top1_tokens[i] in gt_top3_tokens[i]:
                            top1_in_top3_sum += 1


        avg_val_loss = val_loss / val_samples if val_samples > 0 else 0.0
        logger.info(f"Validation loss: {avg_val_loss:.4f}")

        # 计算 top-k 准确率的概率
        top1_in_top1_prob = top1_in_top1_sum / val_samples if val_samples > 0 else 0.0
        top1_in_top2_prob = top1_in_top2_sum / val_samples if val_samples > 0 else 0.0
        top1_in_top3_prob = top1_in_top3_sum / val_samples if val_samples > 0 else 0.0

        logger.info(f"Top-1 in Top-1 Accuracy: {top1_in_top1_prob:.4f}")
        logger.info(f"Top-1 in Top-2 Accuracy: {top1_in_top2_prob:.4f}")
        logger.info(f"Top-1 in Top-3 Accuracy: {top1_in_top3_prob:.4f}")


        # 记录验证 loss 和 top-k 准确率到 wandb
        if accelerator.is_main_process:
            wandb.log({
                "val_loss": avg_val_loss,
                "epoch": epoch + 1,
                "global_step": global_step,
                "top1_in_top1_accuracy": top1_in_top1_prob, # 记录 top1 in top1 准确率
                "top1_in_top2_accuracy": top1_in_top2_prob, # 记录 top1 in top2 准确率
                "top1_in_top3_accuracy": top1_in_top3_prob  # 记录 top1 in top3 准确率
            })

        if global_step >= args.max_train_steps:
            break

    # ========== 保存最终模型 ==========
    if accelerator.is_main_process:
        final_model = accelerator.unwrap_model(model)
        torch.save(final_model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
        logger.info(f"Model saved to {args.output_dir}")

    # 结束 wandb 记录
    if accelerator.is_main_process:
        wandb.finish()

    logger.info("Training & Inference finished.")


if __name__ == "__main__":
    main()