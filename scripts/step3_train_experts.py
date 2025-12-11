import os
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import joblib

# --- Configuration (Relative Paths) ---
# 使用相对路径，确保在任何电脑上都能跑
FEATURE_DIR = "./data/features"
OUTPUT_DIR = "./data/clusters"
NUM_EXPERTS = 4  # 4 experts for demo (20 samples)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_experts():
    print(f"    Starting Tri-Modal Expert Training...")
    print(f"   - Feature Dimension: 1536 (Image+Text+Audio)")
    print(f"   - Number of Experts: {NUM_EXPERTS}")
    print(f"   - Data Source: {FEATURE_DIR}")

    # 1. Initialize MiniBatchKMeans
    # Batch size 256 is standard for large-scale training simulation
    kmeans = MiniBatchKMeans(
        n_clusters=NUM_EXPERTS,
        random_state=42,
        batch_size=256,
        n_init='auto'
    )

    # 2. Load Data
    if not os.path.exists(FEATURE_DIR):
        print(f"  Error: Feature directory not found: {FEATURE_DIR}")
        print("   Please run 'src/prep_feature_tri.py' first.")
        return

    files = [f for f in os.listdir(FEATURE_DIR) if f.endswith('_feat.pth')]
    
    if not files:
        print(f" Error: No .pth feature files found in {FEATURE_DIR}")
        return

    all_features = []

    print(f"  Loading features from {len(files)} files...")
    for f in files:
        path = os.path.join(FEATURE_DIR, f)
        try:
            data = torch.load(path)
            # data['feat'] is Tensor, convert to numpy for sklearn
            feats = data['feat'].numpy()
            all_features.append(feats)
        except Exception as e:
            print(f" Warning: Failed to load {f}: {e}")

    # Concatenate all features
    X = np.concatenate(all_features, axis=0)
    print(f" Data Loaded. Total Samples: {X.shape[0]}, Dimensions: {X.shape[1]}")

    # 3. Training (Clustering)
    print(" Running Mini-Batch K-Means Clustering...")
    kmeans.fit(X)

    # 4. Save Expert Centers
    # Save as PyTorch tensor for easier integration in the inference step
    centers_tensor = torch.from_numpy(kmeans.cluster_centers_)
    save_path = os.path.join(OUTPUT_DIR, f"experts_c{NUM_EXPERTS}.pth")

    torch.save({
        "centers": centers_tensor,
        "n_experts": NUM_EXPERTS,
        "dim": X.shape[1]
    }, save_path)

    # Save sklearn model as backup
    joblib.dump(kmeans, save_path.replace('.pth', '.pkl'))

    print("-" * 50)
    print(f"  Training Complete!")
    print(f"  Expert Weights Saved: {save_path}")
    print(f"  (Contains {NUM_EXPERTS} Tri-Modal Expert Centroids)")
    print("-" * 50)

    # Simple Analysis
    labels = kmeans.labels_
    print("  Cluster Distribution Preview:")
    for i in range(NUM_EXPERTS):
        count = np.sum(labels == i)
        percentage = (count / len(labels)) * 100
        bar = "█" * int(percentage / 5)
        print(f"   Expert {i}: {count:3d} samples ({percentage:5.1f}%) {bar}")

if __name__ == "__main__":
    train_experts()
