import os
import torch
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# --- 配置 ---
FEATURE_DIR = "D:/TriMode_Data/features"
CLUSTER_PATH = "D:/TriMode_Data/clusters/experts_c4.pth"
OUTPUT_DIR = "D:/TriMode_Data/assignments"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def visualize_tsne(features, labels, centers, save_path):
    """
    生成一张高大上的 t-SNE 聚类分布图，用于展示
    """
    print("正在生成 t-SNE 可视化图...")

    # 合并样本和中心点一起降维
    all_data = np.concatenate([features, centers], axis=0)
    n_samples = features.shape[0]
    n_centers = centers.shape[0]

    # 设置 perplexity (必须小于样本数)
    perp = min(5, n_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='random')
    embedded = tsne.fit_transform(all_data)

    # 拆分
    X_embedded = embedded[:n_samples]
    C_embedded = embedded[n_samples:]

    plt.figure(figsize=(10, 8), dpi=120)

    # 画样本点
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='viridis', alpha=0.6, s=100, label='Data')

    # 画中心点
    plt.scatter(C_embedded[:, 0], C_embedded[:, 1], c='red', marker='X', s=300, edgecolor='black',
                label='Expert Centers')

    plt.title("Tri-Modal Data Experts Distribution (t-SNE)")
    plt.legend()
    plt.colorbar(scatter, label='Expert ID')
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path)
    plt.close()
    print(f"图片已保存: {save_path}")


def assign_and_visualize():
    print("开始执行专家指派 (Routing)...")

    # 1. 加载专家权重
    if not os.path.exists(CLUSTER_PATH):
        print(f"找不到专家文件: {CLUSTER_PATH}")
        return

    checkpoint = torch.load(CLUSTER_PATH)
    centers = checkpoint['centers']  # Tensor [4, 1536]
    print(f"加载了 {centers.shape[0]} 个专家中心。")

    # 2. 加载数据
    files = [f for f in os.listdir(FEATURE_DIR) if f.endswith('_feat.pth')]
    all_feats = []
    all_keys = []

    for f in files:
        path = os.path.join(FEATURE_DIR, f)
        data = torch.load(path)
        all_feats.append(data['feat'])
        all_keys.extend(data['filekeys'])

    X = torch.cat(all_feats, dim=0)  # [N, 1536]

    # 3. 计算指派 (Assignment)
    # 计算每个样本到 4 个中心的距离
    # cdist 支持 Euclidean 距离
    dists = torch.cdist(X, centers)

    # 取最近的中心作为 label
    min_dists, labels = dists.min(dim=1)

    # 4. 保存结果 (JSON)
    results = {}
    for i, key in enumerate(all_keys):
        expert_id = int(labels[i])
        results[key] = expert_id

    json_path = os.path.join(OUTPUT_DIR, "final_assignments.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"指派完成！结果已保存至: {json_path}")

    # 5. 可视化 (加分项)
    # 生成一张图，直观展示聚类效果
    vis_path = os.path.join(OUTPUT_DIR, "experts_visualization.png")
    visualize_tsne(X.numpy(), labels.numpy(), centers.numpy(), vis_path)

    print("-" * 50)
    print("Tri-MoDE 全流程执行完毕！")
    print("已生成：")
    print("1. 原始三模态数据 (wav/txt/jpg)")
    print("2. 融合特征向量 (1536维)")
    print("3. 数据专家系统 (4个专家)")
    print("4. 可视化证明图 (png)")
    print("-" * 50)


if __name__ == "__main__":
    assign_and_visualize()