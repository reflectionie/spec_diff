#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
此脚本用于遍历指定目录下的所有文件，读取每个文件中的真实 hidden state（gt/hidden_state）
和草稿 hidden state（draft_hidden/draft），分别计算两者的均值和标准差，用于归一化。
文件处理部分使用多线程加速。

用法示例：
    python compute_hidden_stats_mt.py --data_dir /path/to/data --num_workers 40
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
    加载单个文件，分别提取真实 hidden state（gt）和草稿 hidden state（draft），
    对各自展平后计算其和、平方和以及元素个数。
    若文件无法加载或格式不符合要求，则返回 None。
    
    返回值为一个六元组：
      (gt_sum, gt_sum_sq, gt_count, draft_sum, draft_sum_sq, draft_count)
    """
    try:
        data = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"加载文件 {path} 时出错：{e}")
        return None

    # 获取真实 hidden state（gt）的数据
    gt = None
    if "hidden_state" in data:
        gt = data["hidden_state"]
    elif "gt" in data:
        gt = data["gt"]

    # 获取草稿 hidden state（draft）的数据
    draft = None
    if "draft_hidden" in data:
        draft = data["draft_hidden"]
    elif "draft" in data:
        draft = data["draft"]

    if gt is None and draft is None:
        print(f"文件 {path} 中未找到 'hidden_state'/'gt' 和 'draft_hidden'/'draft' 键，跳过。")
        return None

    # 处理 gt 数据（如果存在）
    if gt is not None:
        # 尝试 squeeze 第一维（例如 [1, seq_len, hidden_dim] -> [seq_len, hidden_dim]）
        if gt.dim() == 3 and gt.size(0) == 1:
            gt = gt.squeeze(0)
        gt_flat = gt.flatten().to(torch.float64)
        gt_sum = gt_flat.sum().item()
        gt_sum_sq = (gt_flat ** 2).sum().item()
        gt_count = gt_flat.numel()
    else:
        gt_sum, gt_sum_sq, gt_count = 0.0, 0.0, 0

    # 处理 draft 数据（如果存在）
    if draft is not None:
        if draft.dim() == 3 and draft.size(0) == 1:
            draft = draft.squeeze(0)
        draft_flat = draft.flatten().to(torch.float64)
        draft_sum = draft_flat.sum().item()
        draft_sum_sq = (draft_flat ** 2).sum().item()
        draft_count = draft_flat.numel()
    else:
        draft_sum, draft_sum_sq, draft_count = 0.0, 0.0, 0

    return gt_sum, gt_sum_sq, gt_count, draft_sum, draft_sum_sq, draft_count

def main():
    parser = argparse.ArgumentParser(
        description="分别计算所有文件中真实 hidden state（gt）和草稿 hidden state（draft）的均值和标准差（使用多线程处理）")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="存放包含 hidden state 文件的文件夹")
    parser.add_argument("--num_workers", type=int, default=40,
                        help="线程数（默认为40）")
    args = parser.parse_args()

    file_paths = list_files(args.data_dir)
    if not file_paths:
        print("未在指定目录下找到任何文件。")
        return

    # 全局统计变量
    total_gt_sum = 0.0
    total_gt_sum_sq = 0.0
    total_gt_count = 0

    total_draft_sum = 0.0
    total_draft_sum_sq = 0.0
    total_draft_count = 0

    valid_files = 0

    # 使用多线程并行处理文件
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_path = {executor.submit(process_file, path): path for path in file_paths}
        for future in tqdm(as_completed(future_to_path), total=len(file_paths), desc="处理文件", ncols=80):
            result = future.result()
            if result is None:
                continue
            gt_sum, gt_sum_sq, gt_count, draft_sum, draft_sum_sq, draft_count = result

            total_gt_sum += gt_sum
            total_gt_sum_sq += gt_sum_sq
            total_gt_count += gt_count

            total_draft_sum += draft_sum
            total_draft_sum_sq += draft_sum_sq
            total_draft_count += draft_count

            valid_files += 1

    if valid_files == 0:
        print("没有有效数据被处理！")
        return

    print("=" * 40)
    print(f"共处理文件数：{valid_files} (总文件数：{len(file_paths)})")

    if total_gt_count > 0:
        mean_gt = total_gt_sum / total_gt_count
        var_gt = total_gt_sum_sq / total_gt_count - mean_gt ** 2
        std_gt = var_gt ** 0.5
        print(f"真实 hidden state 的处理元素数：{total_gt_count}")
        print(f"真实 hidden state 的均值：{mean_gt:.6f}")
        print(f"真实 hidden state 的标准差：{std_gt:.6f}")
    else:
        print("未处理到任何真实 hidden state 数据。")

    print("-" * 40)

    if total_draft_count > 0:
        mean_draft = total_draft_sum / total_draft_count
        var_draft = total_draft_sum_sq / total_draft_count - mean_draft ** 2
        std_draft = var_draft ** 0.5
        print(f"草稿 hidden state 的处理元素数：{total_draft_count}")
        print(f"草稿 hidden state 的均值：{mean_draft:.6f}")
        print(f"草稿 hidden state 的标准差：{std_draft:.6f}")
    else:
        print("未处理到任何草稿 hidden state 数据。")
    print("=" * 40)

if __name__ == "__main__":
    main()