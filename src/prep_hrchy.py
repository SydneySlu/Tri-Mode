# Copyright (c) Meta Platforms, Inc. and affiliates

import sys

sys.path.append("src")
sys.path.append("./")
import os
import random

import torch
import torch.nn.functional as F
import argparse, pickle, pdb
import numpy as np

from kmeans_pytorch import KMeans as BalancedKMeans
from get_prep_parser import get_args_parser, get_default_paths

# 自动检测设备 (CPU用户的福音)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    print("⚠️ 未检测到 CUDA，正在使用 CPU 运行聚类...")


def cluster_fine(n_clusters, args, balanced=1):
    path_to_fine = os.path.join(args.ccenter_dir, args.cd, f'F{n_clusters}.pth')
    os.makedirs(os.path.dirname(path_to_fine), exist_ok=True)
    if os.path.exists(path_to_fine):
        print(f'{path_to_fine} is written')
        return True

    print(f'Preparing data for file {path_to_fine}')
    file_for_run = []

    # --- 修改点 1：安全遍历文件夹 ---
    for i in range(100):
        # 构造路径
        dir_path = os.path.join(args.feature_dir, str(i))

        # 如果文件夹不存在，直接跳过 (防止 67, 68... 报错)
        if not os.path.exists(dir_path):
            continue

        feat_files = os.listdir(dir_path)
        num_files = len(feat_files)

        # 如果是空文件夹，也跳过
        if num_files == 0:
            continue

        num_fun_files = int(num_files * 0.05) + 1  # num_files
        files = np.random.choice(feat_files, num_fun_files).tolist()
        file_for_run.extend([os.path.join(dir_path, file) for file in files])
    # ---------------------------

    if len(file_for_run) == 0:
        print("❌ 错误：没有找到任何特征文件！请检查 prep_feature.py 是否成功生成了 .pth 文件。")
        return False

    np.random.shuffle(file_for_run)
    print('{} files are selected'.format(len(file_for_run)))

    # 聚类器初始化
    kmeans = BalancedKMeans(n_clusters=n_clusters, device=device, balanced=(balanced == 1))
    total_size = 0

    for i, file in enumerate(file_for_run):
        print(i, file)
        # --- 修改点 2：将 .cuda() 改为 .to(device) ---
        # 还要确保加载时 map_location 指向设备
        data = torch.load(file, map_location=device)
        feat = F.normalize(data['feat'].to(device), dim=-1)

        total_size += feat.size(0)

        # 注意：kmeans.fit 内部可能也有 device 处理，但输入必须对齐
        if 'cos' in args.cd:
            kmeans.fit(feat, distance='cosine', iter_limit=50, online=True, iter_k=i)
        elif 'euc' in args.cd.lower():  # euclidean
            kmeans.fit(feat, distance='euclidean', iter_limit=50, online=True, iter_k=i)
        else:
            raise ValueError('Not Implemented')

        if (i + 1) % 100 == 0:
            print(f'checkpointing at step {i}')
            # 保存时转回 CPU 以便兼容
            torch.save({'center': kmeans.cluster_centers.cpu()}, path_to_fine)

    print('there are {} files involved in clustering'.format(total_size))

    # 保存结果
    try:
        with open(path_to_fine.replace('.pth', '.pkl'), 'wb+') as f:
            _ = pickle.dump(kmeans, f)
    except Exception as e:
        print(f"Warning: Pickle dump failed ({e}), skipping pkl save.")

    torch.save({'center': kmeans.cluster_centers.cpu()}, path_to_fine)
    print(f"✅ 成功生成数据专家权重: {path_to_fine}")
    return True


def cluster_coarse(n_clusters, args, balanced=1):
    path_to_fine = os.path.join(args.ccenter_dir, args.cd, f'F{args.cm}.pth')
    if not os.path.exists(path_to_fine):
        print(f"❌ 找不到细粒度中心文件: {path_to_fine}，无法进行粗粒度聚类。")
        return False

    centers = torch.load(path_to_fine, map_location=device)['center']

    path_to_coarse = os.path.join(args.ccenter_dir, args.cd, f'F{args.cm}-C{n_clusters}.pth')
    if os.path.exists(path_to_coarse):
        print(f'{path_to_coarse} is written')
        return True

    kmeans = BalancedKMeans(n_clusters=n_clusters, device=device, balanced=(balanced == 1))

    # --- 修改点 3：将 .cuda() 改为 .to(device) ---
    if 'cos' in args.cd:
        kmeans.fit(F.normalize(centers.to(device), dim=-1), distance='cosine', iter_limit=100, online=False)
    elif 'euc' in args.cd.lower():  # euclidean
        kmeans.fit(centers.to(device), distance='euclidean', iter_limit=100, online=False)
    else:
        raise ValueError('Not Implemented')

    assign = kmeans.predict(centers.to(device), args.cd)
    torch.save({'coarse': kmeans.cluster_centers.cpu(), 'assign': assign.cpu()}, path_to_coarse)
    print(f"✅ 成功生成粗粒度聚类: {path_to_coarse}")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Clustering Evaluation', parents=[get_args_parser()])
    config = parser.parse_args()

    # 确保输出目录存在
    paths = get_default_paths()[config.dataset]
    config.feature_dir = paths['feature']
    config.ccenter_dir = paths['cluster']

    # 运行聚类
    cluster_fine(config.cm, config)
    cluster_coarse(config.cn, config)