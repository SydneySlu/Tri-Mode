# Copyright (c) Meta Platforms, Inc. and affiliates
# Modified for Tri-Modal Project

import sys
sys.path.append("src")
sys.path.append("./")

import os
import torch
import torch.nn.functional as F
import argparse
import json
from multiprocessing import Pool
from tqdm import tqdm

from get_prep_parser import get_args_parser, get_default_paths
from prep_caption import get_tarfile_path

# --- Worker Function (Executed in Sub-Processes) ---
def build_assignment(feat_dir, shard_id, assign_dir, ccenters, overwrite=True):
    """
    Computes distance between sample features and expert centroids.
    """
    # Create sub-folder structure (0/00000000...)
    shard_folder = str(shard_id % 100)
    output_fn = os.path.join(assign_dir, shard_folder, f'{shard_id}_assign_dist.json')
    os.makedirs(os.path.dirname(output_fn), exist_ok=True)

    # Construct feature file path
    # Note: In Tri-MoDE, features are flat in ./data/features/, not nested in 0/1/2
    # So we check both locations for compatibility
    feat_fn_nested = os.path.join(feat_dir, shard_folder, f'{shard_id:08d}_feat.pth')
    feat_fn_flat = os.path.join(feat_dir, f'{shard_id:08d}_feat.pth')
    
    if os.path.exists(feat_fn_nested):
        feat_fn = feat_fn_nested
    elif os.path.exists(feat_fn_flat):
        feat_fn = feat_fn_flat
    else:
        # print(f"[Skip] Feature file not found for shard {shard_id}")
        return None

    if os.path.exists(output_fn) and not overwrite:
        return True

    try:
        # Force CPU load
        feature = torch.load(feat_fn, map_location='cpu')
        
        # Prepare output structure
        assign = {'key': feature['filekeys'], 'image': feature['img_midx']}
        
        # Normalize features
        feat_tensor = F.normalize(feature['feat'], dim=-1)

        # Iterate over cluster centers (e.g., Euclidean, Cosine)
        for key, ccenter in ccenters.items():
            # Distance Calculation
            if key.startswith('E'):  # Euclidean
                # cdist requires float type consistency
                dist = torch.cdist(feat_tensor.float()[None], ccenter.float()[None])[0]
                min_dist, assign_tensor = dist.min(dim=-1)
            elif key.startswith('C'):  # Cosine
                sim = torch.mm(feat_tensor.float(), ccenter.T.float())
                max_sim, assign_tensor = sim.max(dim=-1)
            
            # Store assignments
            assign[key] = {'assign': assign_tensor.numpy().tolist()}

        # Save to JSON
        with open(output_fn, 'w') as json_file:
            json.dump(assign, json_file)
        
        return assign

    except Exception as e:
        print(f"[Error] Failed to process shard {shard_id}: {e}")
        return None


def worker_routine(args, task_ids):
    """
    Worker process entry point.
    Handles loading centroids independently to avoid Windows multiprocessing pickle errors.
    """
    # --- 1. Load Cluster Centers (Independent per process) ---
    local_ccenters = {}
    
    # We only use Euclidean for this project demo
    dist_type = 'euclidean'
    cm = args.cm
    
    # Path construction
    center_path = os.path.join(args.ccenter_dir, dist_type, f'F{cm}.pth')
    
    if os.path.exists(center_path):
        key = f'{dist_type[0].upper()}{cm}' # e.g., E4
        data = torch.load(center_path, map_location='cpu')
        local_ccenters[key] = data['center']
        
        if 'cos' in dist_type:
            local_ccenters[key] = F.normalize(local_ccenters[key], dim=-1)
    else:
        # Try flat directory structure as fallback
        center_path_flat = os.path.join(args.ccenter_dir, f'experts_c{cm}.pth') # From step3 script
        if os.path.exists(center_path_flat):
             key = f'E{cm}'
             data = torch.load(center_path_flat, map_location='cpu')
             # step3 saves 'centers', prep_hrchy saves 'center'
             if 'centers' in data:
                 local_ccenters[key] = data['centers']
             else:
                 local_ccenters[key] = data['center']
        else:
             print(f"[Error] Worker could not find center file: {center_path}")
             return []

    # --- 2. Process Shards ---
    missing_shards = []
    
    # Use tqdm only for the first worker to avoid console spam
    iterator = task_ids
    if len(task_ids) > 0 and task_ids[0] == 0:
        iterator = tqdm(task_ids, desc="[Inference]")

    for shard_id in iterator:
        status = build_assignment(
            args.feature_dir, shard_id, args.cassign_dir, local_ccenters, overwrite=False
        )
        if status is None:
            missing_shards.append(shard_id)

    return missing_shards


def main(args):
    print(">>> Starting Expert Inference (Routing)...")
    print(f"[Config] Range: {args.tar_init} -> {args.tar_end}")
    print(f"[Config] Clusters (cm): {args.cm}")
    print(f"[Config] Feature Dir: {args.feature_dir}")
    print(f"[Config] Output Dir: {args.cassign_dir}")

    # Task Distribution
    # Create a list of shard IDs to process
    # +1 to include the end index
    all_shards = list(range(args.tar_init, args.tar_end + 1))
    
    # Split shards into chunks for workers
    chunk_size = len(all_shards) // args.num_threads + 1
    tasks = [all_shards[i:i + chunk_size] for i in range(0, len(all_shards), chunk_size)]
    
    # Ensure task list matches thread count (remove empty lists)
    tasks = [t for t in tasks if len(t) > 0]
    
    print(f"[Info] Dispatching {len(all_shards)} tasks to {len(tasks)} workers...")

    # Argument preparation for starmap
    # Note: We pass args, but workers will load their own ccenters
    worker_args = [(args, task_chunk) for task_chunk in tasks]

    # Start Multiprocessing Pool
    with Pool(processes=len(tasks)) as p:
        results = p.starmap(worker_routine, worker_args)

    # Aggregate results
    all_missing = []
    for res in results:
        all_missing.extend(res)
    
    print("-" * 60)
    print(f"[Success] Inference Complete.")
    if len(all_missing) > 0:
        print(f"[Warning] {len(all_missing)} shards were missing or failed.")
    print("-" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Inference Evaluation', parents=[get_args_parser()])
    config = parser.parse_args()

    # --- Override Paths for Tri-MoDE ---
    if not config.feature_dir:
        config.feature_dir = "./data/features"
    if not config.cassign_dir:
        config.cassign_dir = "./data/assignments"
    if not config.ccenter_dir:
        config.ccenter_dir = "./data/cluster_center"

    # --- Auto-detect Range ---
    # In our project, we usually have shard 0 to 0 (for demo) or 0 to 19 (for full)
    # Let's try to detect the max shard ID from the feature directory
    if config.tar_end == -1:
        try:
            files = [f for f in os.listdir(config.feature_dir) if f.endswith('_feat.pth')]
            if files:
                # Extract numbers from "00000000_feat.pth"
                ids = [int(f.split('_')[0]) for f in files]
                config.tar_init = min(ids)
                config.tar_end = max(ids)
                print(f"[Config] Auto-detected shard range: {config.tar_init}..{config.tar_end}")
            else:
                config.tar_end = 0 # Default fallback
        except Exception as e:
            print(f"[Warning] Failed to auto-detect range: {e}. Using default 0.")
            config.tar_end = 0

    config.num_threads = 4  # Set to 4 for safety on most CPUs

    os.makedirs(config.cassign_dir, exist_ok=True)

    main(config)
