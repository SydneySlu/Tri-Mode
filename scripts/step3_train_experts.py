import os
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import joblib

# --- 配置 ---
FEATURE_DIR = "D:/TriMode_Data/features"
OUTPUT_DIR = "D:/TriMode_Data/clusters"
NUM_EXPERTS = 4  # 因为我们只有 20 个样本，设 4 个专家比较合理 (每个专家分 5 个)
# 真实场景下，如果有 100万数据，这里可以设为 64 或 1024

os.makedirs(OUTPUT_DIR, exist_ok=True)


def train_experts():
    print("开始训练三模态数据专家 (Tri-Modal Experts)...")
    print(f"特征维度: 1536 (Image+Text+Audio)")
    print(f"专家数量: {NUM_EXPERTS}")

    # 1. 初始化 MiniBatchKMeans
    # batch_size=256 是典型设置，模拟大规模训练
    kmeans = MiniBatchKMeans(
        n_clusters=NUM_EXPERTS,
        random_state=42,
        batch_size=256,
        n_init='auto'
    )

    # 2. 加载数据
    # 在真实大规模场景下，我们会写一个生成器(Generator)分批读取
    # 这里数据量小，我们一次性读取演示原理
    files = [f for f in os.listdir(FEATURE_DIR) if f.endswith('_feat.pth')]
    all_features = []

    print("正在加载特征数据...")
    for f in files:
        path = os.path.join(FEATURE_DIR, f)
        data = torch.load(path)
        # data['feat'] 是 Tensor，转为 numpy
        feats = data['feat'].numpy()
        all_features.append(feats)

    if not all_features:
        print("错误：没有找到特征文件！")
        return

    # 拼接所有数据
    X = np.concatenate(all_features, axis=0)
    print(f"加载完成，总样本数: {X.shape[0]}, 特征维度: {X.shape[1]}")

    # 3. 训练 (Clustering)
    print("正在进行聚类训练 (模拟大规模 Mini-Batch 更新)...")
    kmeans.fit(X)

    # 4. 保存专家中心 (Center)
    # 我们存成 PyTorch 格式，方便后续 torch 计算
    centers_tensor = torch.from_numpy(kmeans.cluster_centers_)
    save_path = os.path.join(OUTPUT_DIR, f"experts_c{NUM_EXPERTS}.pth")

    torch.save({
        "centers": centers_tensor,
        "n_experts": NUM_EXPERTS,
        "dim": X.shape[1]
    }, save_path)

    # 顺便保存 sklearn 模型，方便以后复用
    joblib.dump(kmeans, save_path.replace('.pth', '.pkl'))

    print("-" * 50)
    print(f"训练完成")
    print(f"专家权重已保存: {save_path}")
    print(f"(包含 {NUM_EXPERTS} 个三模态专家的核心特征)")
    print("-" * 50)

    # 简单分析一下
    labels = kmeans.labels_
    print("初步聚类分布预览:")
    for i in range(NUM_EXPERTS):
        count = np.sum(labels == i)
        print(f"   Expert {i}: {count} 个样本")


if __name__ == "__main__":
    train_experts()