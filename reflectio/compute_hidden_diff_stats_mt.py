#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
此脚本用于遍历指定目录下的所有文件，读取每个文件中的草稿 hidden state（draft_hidden）和真实 hidden state（hidden_state），
并计算所有 hidden state 对应位置上 (gt - draft) 差值的均值和标准差。文件处理部分使用多线程加速。

用法示例：
    python compute_hidden_diff_stats_mt.py --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data --num_workers 40

"""

import os
import argparse
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def list_files(path):
    """
    遍历指定目录，返回所有文件的完整路径列表
    """
    file_list = []
    for root, _, files in os.walk(path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def process_file(path):
    """
    加载单个文件，计算 (gt - draft) 的和、平方和以及元素个数。
    若文件无法加载或格式不符合要求，则返回 None
    """
    try:
        data = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"加载文件 {path} 时出错：{e}")
        return None

    # 尝试获取草稿与真实 hidden state 的数据
    if "draft_hidden" in data:
        draft = data["draft_hidden"]
    elif "draft" in data:
        draft = data["draft"]
    else:
        print(f"文件 {path} 中未找到 'draft_hidden' 或 'draft' 键，跳过。")
        return None

    if "hidden_state" in data:
        gt = data["hidden_state"]
    elif "gt" in data:
        gt = data["gt"]
    else:
        print(f"文件 {path} 中未找到 'hidden_state' 或 'gt' 键，跳过。")
        return None

    # 尝试 squeeze 第一维（例如 [1, seq_len, hidden_dim] -> [seq_len, hidden_dim]）
    if draft.dim() == 3 and draft.size(0) == 1:
        draft = draft.squeeze(0)
    if gt.dim() == 3 and gt.size(0) == 1:
        gt = gt.squeeze(0)

    if draft.shape != gt.shape:
        print(f"文件 {path} 中 draft 与 gt 的形状不匹配：{draft.shape} vs {gt.shape}，跳过。")
        return None

    # 计算差值：gt - draft
    diff = gt - draft
    diff_flat = diff.flatten().to(torch.float64)

    file_sum = diff_flat.sum().item()
    file_sum_sq = (diff_flat ** 2).sum().item()
    file_count = diff_flat.numel()

    return file_sum, file_sum_sq, file_count

def main():
    parser = argparse.ArgumentParser(
        description="计算所有文件中 (gt - draft) 的 hidden state 差值的均值和标准差（使用多线程处理）")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="存放包含 hidden state 文件的文件夹")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="线程数（默认为8）")
    args = parser.parse_args()

    file_paths = list_files(args.data_dir)
    if not file_paths:
        print("未在指定目录下找到任何文件。")
        return

    total_sum = 0.0       # 差值之和
    total_sum_sq = 0.0    # 差值平方和
    total_count = 0       # 总元素个数
    valid_files = 0       # 成功处理的文件数

    # 使用多线程并行处理文件
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_path = {executor.submit(process_file, path): path for path in file_paths}
        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(future_to_path), total=len(file_paths), desc="处理文件", ncols=80):
            result = future.result()
            if result is None:
                continue
            file_sum, file_sum_sq, file_count = result
            total_sum += file_sum
            total_sum_sq += file_sum_sq
            total_count += file_count
            valid_files += 1

    if total_count == 0:
        print("没有有效数据被处理！")
        return

    mean_diff = total_sum / total_count
    var_diff = total_sum_sq / total_count - mean_diff ** 2
    std_diff = var_diff ** 0.5

    print("=" * 40)
    print(f"共处理文件数：{valid_files} (总文件数：{len(file_paths)})")
    print(f"处理元素总数：{total_count}")
    print(f"(gt - draft) 的均值：{mean_diff:.6f}")
    print(f"(gt - draft) 的标准差：{std_diff:.6f}")
    print("=" * 40)

if __name__ == "__main__":
    main()
