# Copyright (c) Meta Platforms, Inc. and affiliates
# Modified for Tri-Modal Project

import sys
sys.path.append("src")
sys.path.append("./")

import os
import random
import torch
import torch.nn.functional as F
import argparse
import pickle
import numpy as np

# Dependency: pip install kmeans-pytorch
from kmeans_pytorch import KMeans as BalancedKMeans
from get_prep_parser import get_args_parser, get_default_paths

# --- Device Detection ---
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("[Config] CUDA detected. Using GPU for clustering.")
else:
    device = torch.device('cpu')
    print("[Config] CUDA not found. Switching to CPU mode.")


def cluster_fine(n_clusters, args, balanced=1):
    path_to_fine = os.path.join(args.ccenter_dir, args.cd, f'F{n_clusters}.pth')
    os.makedirs(os.path.dirname(path_to_fine), exist_ok=True)
    
    if os.path.exists(path_to_fine):
        print(f'[Info] File already exists, skipping: {path_to_fine}')
        return True

    print(f'[Info] Preparing data for fine clustering: {path_to_fine}')
    file_for_run = []

    # --- Robust Directory Traversal ---
    # Iterating through potential subdirectories (legacy support) or root dir
    search_dirs = [args.feature_dir]
    # Add numbered subdirectories if they exist (0-99)
    for i in range(100):
        sub_dir = os.path.join(args.feature_dir, str(i))
        if os.path.exists(sub_dir):
            search_dirs.append(sub_dir)

    for dir_path in search_dirs:
        if not os.path.exists(dir_path):
            continue
            
        feat_files = [f for f in os.listdir(dir_path) if f.endswith('.pth') and 'feat' in f]
        num_files = len(feat_files)

        if num_files == 0:
            continue

        # Sampling strategy: Take 5% of files to speed up clustering on large datasets
        # For small datasets, take at least 1 file
        num_fun_files = int(num_files * 0.05) + 1 
        selected_files = np.random.choice(feat_files, num_fun_files, replace=False).tolist()
        file_for_run.extend([os.path.join(dir_path, f) for f in selected_files])

    if len(file_for_run) == 0:
        print("[Error] No feature files found! Please check 'prep_feature_tri.py' output.")
        return False

    np.random.shuffle(file_for_run)
    print(f'[Info] Selected {len(file_for_run)} files for clustering training.')

    # --- Load Data into Memory ---
    all_feats = []
    total_size = 0
    
    print("[Info] Loading features...")
    for i, file in enumerate(file_for_run):
        try:
            # Map location ensures CPU compatibility if trained on GPU
            data = torch.load(file, map_location=device)
            
            # Handle different saving formats (dict vs tensor)
            if isinstance(data, dict) and 'feat' in data:
                feat = data['feat']
            else:
                feat = data
            
            # Normalize features
            feat = F.normalize(feat.to(device), dim=-1)
            all_feats.append(feat)
            total_size += feat.size(0)
            
        except Exception as e:
            print(f"[Warning] Failed to load {file}: {e}")

    if not all_feats:
        print("[Error] No valid features loaded.")
        return False

    # Concatenate all features
    X = torch.cat(all_feats, dim=0)
    print(f'[Info] Total samples: {total_size}, Dimension: {X.size(1)}')

    # Safety check: Cannot cluster if samples < n_clusters
    if total_size < n_clusters:
        print(f"[Warning] Sample size ({total_size}) is smaller than target clusters ({n_clusters}).")
        print(f"          Adjusting n_clusters to {total_size}.")
        n_clusters = total_size

    # --- Clustering Execution ---
    print(f'[Info] Initializing BalancedKMeans (K={n_clusters})...')
    kmeans = BalancedKMeans(n_clusters=n_clusters, device=device, balanced=(balanced == 1))

    # Fit logic
    dist_metric = 'cosine' if 'cos' in args.cd else 'euclidean'
    print(f"[Info] Fitting using {dist_metric} distance...")
    
    kmeans.fit(X, distance=dist_metric, iter_limit=50, online=False)

    print('[Success] Clustering converged.')

    # Save sklearn-compatible pickle (optional)
    try:
        pkl_path = path_to_fine.replace('.pth', '.pkl')
        with open(pkl_path, 'wb+') as f:
            pickle.dump(kmeans, f)
    except Exception as e:
        print(f"[Warning] Pickle dump failed ({e}), skipping .pkl save.")

    # Save PyTorch centroids (Critical for next steps)
    torch.save({'center': kmeans.cluster_centers.cpu()}, path_to_fine)
    print(f"[Success] Expert weights saved to: {path_to_fine}")
    return True


def cluster_coarse(n_clusters, args, balanced=1):
    # This step clusters the fine centroids into coarse centroids (Hierarchy)
    path_to_fine = os.path.join(args.ccenter_dir, args.cd, f'F{args.cm}.pth')
    
    if not os.path.exists(path_to_fine):
        print(f"[Error] Fine cluster centers not found: {path_to_fine}")
        print("        Cannot proceed with coarse clustering.")
        return False

    centers = torch.load(path_to_fine, map_location=device)['center']
    print(f"[Info] Loaded {centers.size(0)} fine-grained centers.")

    path_to_coarse = os.path.join(args.ccenter_dir, args.cd, f'F{args.cm}-C{n_clusters}.pth')
    
    # Check if we actually need clustering (if fine <= coarse)
    if centers.size(0) <= n_clusters:
        print(f"[Info] Fine centers ({centers.size(0)}) <= Coarse target ({n_clusters}).")
        print("       Skipping secondary clustering. Using fine centers as coarse.")
        assign = torch.arange(centers.size(0))
        torch.save({'coarse': centers.cpu(), 'assign': assign.cpu()}, path_to_coarse)
        print(f"[Success] Coarse weights saved to: {path_to_coarse}")
        return True

    if os.path.exists(path_to_coarse):
        print(f'[Info] File already exists, skipping: {path_to_coarse}')
        return True

    print(f'[Info] Starting coarse clustering (K={n_clusters})...')
    kmeans = BalancedKMeans(n_clusters=n_clusters, device=device, balanced=(balanced == 1))

    dist_metric = 'cosine' if 'cos' in args.cd else 'euclidean'
    
    if dist_metric == 'cosine':
        centers_input = F.normalize(centers.to(device), dim=-1)
    else:
        centers_input = centers.to(device)

    kmeans.fit(centers_input, distance=dist_metric, iter_limit=100, online=False)

    # Predict assignments
    assign = kmeans.predict(centers_input, choice=dist_metric)
    
    torch.save({'coarse': kmeans.cluster_centers.cpu(), 'assign': assign.cpu()}, path_to_coarse)
    print(f"[Success] Coarse clustering weights saved to: {path_to_coarse}")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Clustering Evaluation', parents=[get_args_parser()])
    config = parser.parse_args()

    # --- Path Configuration (Override for Tri-MoDE structure) ---
    # Default to relative paths if not specified
    if not config.feature_dir:
        config.feature_dir = "./data/features"
    if not config.ccenter_dir:
        config.ccenter_dir = "./data/cluster_center"

    print(f"[Config] Feature Dir: {config.feature_dir}")
    print(f"[Config] Cluster Dir: {config.ccenter_dir}")
    print(f"[Config] Fine Clusters (cm): {config.cm}")
    print(f"[Config] Coarse Clusters (cn): {config.cn}")

    # Execution
    cluster_fine(config.cm, config)
    cluster_coarse(config.cn, config)
