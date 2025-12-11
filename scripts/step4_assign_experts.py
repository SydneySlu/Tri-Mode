import os
import torch
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# --- Configuration (Relative Paths) ---
FEATURE_DIR = "./data/features"
# Note: Ensure the filename matches the output from step3 (e.g., experts_c4.pth)
CLUSTER_PATH = "./data/clusters/experts_c4.pth" 
OUTPUT_DIR = "./data/assignments"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def visualize_tsne(features, labels, centers, save_path):
    """
    Generates a t-SNE clustering distribution plot for visualization.
    """
    print("[Info] Generating t-SNE visualization...")

    # Concatenate samples and centers for dimensionality reduction
    all_data = np.concatenate([features, centers], axis=0)
    n_samples = features.shape[0]
    n_centers = centers.shape[0]

    # Set perplexity (must be less than n_samples)
    perp = min(5, n_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='random')
    embedded = tsne.fit_transform(all_data)

    # Split back into samples and centers
    X_embedded = embedded[:n_samples]
    C_embedded = embedded[n_samples:]

    plt.figure(figsize=(10, 8), dpi=120)

    # Plot data points
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='viridis', alpha=0.6, s=100, label='Data')

    # Plot expert centers
    plt.scatter(C_embedded[:, 0], C_embedded[:, 1], c='red', marker='X', s=300, edgecolor='black',
                label='Expert Centers')

    plt.title("Tri-Modal Data Experts Distribution (t-SNE)")
    plt.legend()
    plt.colorbar(scatter, label='Expert ID')
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path)
    plt.close()
    print(f"[Success] Visualization saved to: {save_path}")


def assign_and_visualize():
    print(">>> Starting Expert Routing process...")

    # 1. Load Expert Weights
    if not os.path.exists(CLUSTER_PATH):
        print(f"[Error] Expert weights file not found: {CLUSTER_PATH}")
        print("       Please run step3_train.py first.")
        return

    checkpoint = torch.load(CLUSTER_PATH)
    centers = checkpoint['centers']  # Tensor [4, 1536]
    print(f"[Info] Loaded {centers.shape[0]} expert centers.")

    # 2. Load Data
    if not os.path.exists(FEATURE_DIR):
        print(f"[Error] Feature directory not found: {FEATURE_DIR}")
        return

    files = [f for f in os.listdir(FEATURE_DIR) if f.endswith('_feat.pth')]
    if not files:
        print(f"[Error] No feature files found in {FEATURE_DIR}")
        return

    all_feats = []
    all_keys = []

    for f in files:
        path = os.path.join(FEATURE_DIR, f)
        try:
            data = torch.load(path)
            all_feats.append(data['feat'])
            all_keys.extend(data['filekeys'])
        except Exception as e:
            print(f"[Warning] Failed to load {f}: {e}")

    X = torch.cat(all_feats, dim=0)  # [N, 1536]

    # 3. Calculate Assignment (Routing)
    # Calculate Euclidean distance from each sample to 4 centers
    dists = torch.cdist(X, centers)

    # Assign to the nearest center (Argmin)
    min_dists, labels = dists.min(dim=1)

    # 4. Save Results (JSON)
    results = {}
    for i, key in enumerate(all_keys):
        expert_id = int(labels[i])
        results[key] = expert_id

    json_path = os.path.join(OUTPUT_DIR, "final_assignments.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"[Success] Assignment complete. Results saved to: {json_path}")

    # 5. Visualization (t-SNE)
    vis_path = os.path.join(OUTPUT_DIR, "experts_visualization.png")
    visualize_tsne(X.numpy(), labels.numpy(), centers.numpy(), vis_path)

    print("-" * 60)
    print(">>> Tri-MoDE Pipeline Execution Complete.")
    print("Generated Artifacts:")
    print("1. Raw Tri-Modal Data (wav/txt/jpg)")
    print("2. Fused Feature Vectors (1536-dim)")
    print("3. Data Expert System (4 Experts)")
    print("4. Visualization Plot (png)")
    print("-" * 60)


if __name__ == "__main__":
    assign_and_visualize()
