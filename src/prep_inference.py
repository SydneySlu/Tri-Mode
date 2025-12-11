# Copyright (c) Meta Platforms, Inc. and affiliates

import sys

sys.path.append("src")
sys.path.append("./")
import os

import torch
import torch.nn.functional as F
import argparse
from get_prep_parser import get_args_parser, get_default_paths
from prep_caption import get_tarfile_path

import json, pdb

from multiprocessing import Pool
from tqdm import tqdm


# --- 修复 1：安全的赋值函数 ---
def build_assignment(feat_dir, shard_id, assign_dir, overwrite=True):
    shard_folder = shard_id % 100
    output_fn_group = os.path.join(assign_dir, f"{shard_folder}", f'{shard_id}_assign_dist.json')
    os.makedirs(os.path.dirname(output_fn_group), exist_ok=True)

    # 构造特征文件路径
    feat_fn = os.path.join(feat_dir, f"{shard_folder}", f'{shard_id}_feat.pth')
    if not os.path.exists(feat_fn):
        # print(f"⚠️ [Skip] 找不到特征文件: {feat_fn}")
        return None

    if os.path.exists(output_fn_group) and not overwrite:
        # print(f'{output_fn_group} Written already')
        return True

    # 强制 CPU 加载
    try:
        feature = torch.load(feat_fn, map_location='cpu')
        assign = {'key': feature['filekeys'], 'image': feature['img_midx']}
        # 归一化特征
        feat_tensor = F.normalize(feature['feat'], dim=-1)

        for key, ccenter in ccenters.items():
            # 计算距离 (CPU)
            if key[0] == 'E':  # euclidean
                # cdist 可能需要 float 类型一致
                dist = torch.cdist(feat_tensor.float()[None], ccenter.float()[None])[0]
                min_dist, assign_tensor = dist.min(dim=-1)
                # min_dist = min_dist.numpy().tolist() # 可选保存距离
            elif key[0] == 'C':  # cosine
                sim = torch.mm(feat_tensor.float(), ccenter.T.float())
                max_sim, assign_tensor = sim.max(dim=-1)
                # min_dist = (1.0 - max_sim).numpy().tolist()

            # 保存分配结果
            assign[key] = {'assign': assign_tensor.numpy().tolist()}

        with open(output_fn_group, 'w') as json_file:
            json.dump(assign, json_file)
        # print(f'✅ 已生成: {shard_id}')
        return assign
    except Exception as e:
        print(f"❌ 处理 {shard_id} 失败: {e}")
        return None


def func(args, _start, _end):
    # --- 修复 2：子进程独立路径解析 ---
    wds_dir = os.path.dirname(args.root)

    # --- 修复 3：子进程独立加载聚类中心 ---
    global ccenters
    ccenters = {}
    for dist_type in ['euclidean']:
        for cm in [args.cm, ]:
            path = os.path.join(args.ccenter_dir, dist_type, f'F{cm}.pth')
            if os.path.exists(path):
                key = '{}{}'.format(dist_type[0].upper(), args.cm)
                # 强制加载到 CPU
                ccenters[key] = torch.load(path, map_location='cpu')['center']
                if 'cos' in dist_type:
                    ccenters[key] = F.normalize(ccenters[key], dim=-1)
            else:
                print(f"❌ 子进程找不到聚类中心文件: {path}")

    missing_shards = []

    # 确定迭代范围
    if isinstance(_start, list):
        warc_iter = _start
    else:
        # 如果是范围，用 tqdm 显示进度
        warc_iter = tqdm(range(_start, _end))

    for idx, shard_id in enumerate(warc_iter):
        # 这里的检查其实是非必须的，因为我们要根据 feature 文件生成 assignment
        # 但保留原逻辑
        wds_fn = get_tarfile_path(wds_dir, shard_id)
        # 兼容性检查：如果 get_tarfile_path 找不到，我们暂时忽略，直接尝试找 feature
        # if not os.path.exists(wds_fn):
        #    continue

        status = build_assignment(
            args.feature_dir, shard_id, args.cassign_dir, overwrite=False,
        )
        if status:
            pass
        elif status is None:
            missing_shards.append(shard_id)
        # else:
        #    raise ValueError('No Implementation Error')

    return missing_shards


def main(args):
    print("✅ 进入 main 函数")

    # --- 修复 4：打印调试信息 ---
    print(f"🔍 目标范围: {args.tar_init} -> {args.tar_end}")
    print(f"🔍 聚类数 (cm): {args.cm}")
    print(f"🔍 特征目录: {args.feature_dir}")
    print(f"🔍 聚类中心目录: {args.ccenter_dir}")

    # 计算任务分配
    shard_ids = [[] for _ in range(args.num_threads)]
    # +1 是因为 range 是左闭右开，我们要包含最后一个文件
    real_end = args.tar_end + 1

    for shard_id in range(args.tar_init, real_end):
        group_offset = shard_id % args.num_threads
        shard_ids[group_offset].append(shard_id)

    print(f"📋 任务分配示例 (线程0): {shard_ids[0][:10]}...")

    starts = shard_ids
    ends = [None for _ in range(len(starts))]
    argss = [args for _ in range(len(starts))]

    # 这里的 wds_dir 在主进程其实没用，主要看 func 里的
    global wds_dir
    wds_dir = os.path.dirname(args.root)

    # 启动多进程
    with Pool(len(starts)) as p:
        results = p.starmap(
            func,
            zip(
                argss,
                starts,
                ends,
            ),
        )

    all_results = []
    for result in results:
        all_results.extend(result)
    print("missing npy count:", len(all_results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Clustering evaluation', parents=[get_args_parser()])
    config = parser.parse_args()

    # 加载默认路径
    if config.dataset in get_default_paths():
        paths = get_default_paths()[config.dataset]
        config.root = paths['root']
        config.feature_dir = paths['feature']
        config.cassign_dir = paths['assign']
        config.ccenter_dir = paths['cluster']

    # --- 修复 5：关键！解析文件范围 ---
    # 如果没指定 tar_end，尝试从 root 路径解析 {00..67}
    if config.tar_end == -1:
        try:
            base = os.path.basename(config.root)
            if '{' in base and '}' in base:
                parts = base.split("{")[1].split("}")[0].split("..")
                config.tar_end = int(parts[1])
                print(f"🔍 自动解析 tar_end = {config.tar_end}")
            else:
                print("⚠️ 警告: 无法从路径解析 tar_end，将默认为 -1 (不执行)")
        except Exception as e:
            print(f"❌ 解析路径范围失败: {e}")

    config.num_threads = 8  # 稍微降低线程数，防止卡顿

    os.makedirs(config.cassign_dir, exist_ok=True)

    print("🚀 开始运行专家指派...")
    main(config)