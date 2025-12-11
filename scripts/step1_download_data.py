import os

# --- 1. 强制镜像加速 ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
print("镜像加速已开启：https://hf-mirror.com")

from datasets import load_dataset, Audio
import soundfile as sf
import numpy as np
import io  # <--- 新增：用于处理二进制流

# 设置保存路径
save_dir = "D:/TriMode_Data/raw"
os.makedirs(save_dir, exist_ok=True)

print("准备下载 ESC-50 数据集...")

try:
    # --- 2. 加载数据集 ---
    # streaming=False 保证先下载完文件
    dataset = load_dataset("ashraq/esc50", split="train", streaming=False)

    print("元数据加载成功！")
    print("正在执行关键操作：禁止自动解码 (Bypass torchcodec)...")

    # 核心修改：decode=False
    # datasets："别尝试解码音频，直接给我二进制 Bytes"
    dataset = dataset.cast_column("audio", Audio(decode=False))

    print(f"已准备就绪,数据集包含 {len(dataset)} 个样本。")
    print("开始提取前 20 个样本到本地...")

    count = 0

    # 遍历数据集
    for i in range(len(dataset)):
        if count >= 20: break

        try:
            sample = dataset[i]

            # --- 3. 手动解码 (绕过 Windows 限制) ---
            # 因为设置了 decode=False，这里拿到的是字典：{'bytes': b'...', 'path': ...}
            audio_bytes = sample['audio']['bytes']
            label = sample['category']

            # 使用 soundfile + io 直接从内存读取二进制数据
            # 这样就彻底不需要 torchcodec 了！
            audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))

            # --- 4. 构造文本 ---
            caption = f"A sound of {label}."

            print(f"\n[样本 {count}]")
            print(f"  - 类别: {label}")
            print(f"  - 音频形状: {audio_array.shape}, SR: {sampling_rate}")

            # --- 5. 保存 ---
            filename = os.path.join(save_dir, f"sample_{count}.wav")
            sf.write(filename, audio_array, sampling_rate)

            with open(filename.replace('.wav', '.txt'), 'w', encoding='utf-8') as f:
                f.write(caption)

            print(f"  - 已保存: {filename}")
            count += 1

        except Exception as inner_e:
            print(f"跳过样本 {i}: {inner_e}")
            continue

    print("-" * 50)
    print(f"数据准备 Phase 1 完成！")
    print(f"数据已保存在: {save_dir}")
    print("-" * 50)

except Exception as e:
    print(f"\n错误: {e}")