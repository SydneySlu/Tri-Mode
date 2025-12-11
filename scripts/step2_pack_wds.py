import os
import tarfile
import json
import librosa
import numpy as np
import matplotlib.pyplot as plt
import io
from tqdm import tqdm

# --- 配置路径 ---
SOURCE_DIR = "./data/raw"  # 刚才下载的原始文件
DEST_DIR = "./data/shards"  # 打包后的输出目录
SAMPLES_PER_TAR = 50  # 每个 tar 包放多少个样本

os.makedirs(DEST_DIR, exist_ok=True)


def create_spectrogram_image(audio_path):
    """
    核心创新点：将音频模态转换为视觉模态 (Mel Spectrogram)
    这样我们就有了真实的"图像"输入，实现了三模态对齐。
    """
    # 1. 读取音频
    y, sr = librosa.load(audio_path, sr=16000)

    # 2. 计算梅尔频谱
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # 3. 画图并保存到内存 (Bytes)
    plt.figure(figsize=(2.24, 2.24), dpi=100)  # 生成适合 CLIP 输入的尺寸
    plt.axis('off')  # 去掉坐标轴
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # 铺满全图
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')

    buf = io.BytesIO()
    plt.savefig(buf, format='jpg', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return buf.read()


def pack_data():
    print(f"开始构建三模态 WebDataset 数据包...")
    print(f"源目录: {SOURCE_DIR}")
    print(f"目标目录: {DEST_DIR}")

    # 1. 收集所有 wav 文件
    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.wav')]
    files.sort()  # 排序保证顺序

    if len(files) == 0:
        print("错误: 源目录里没有 .wav 文件，请检查 step1 是否成功。")
        return

    print(f"发现 {len(files)} 个样本，准备打包...")

    tar_count = 0
    tar_handle = None

    # 使用 tqdm 显示进度
    for idx, wav_file in enumerate(tqdm(files, desc="Processing")):
        # 每 SAMPLES_PER_TAR 个样本换一个新的 tar 包
        if idx % SAMPLES_PER_TAR == 0:
            if tar_handle:
                tar_handle.close()

            tar_name = os.path.join(DEST_DIR, f"{tar_count:08d}.tar")
            tar_handle = tarfile.open(tar_name, "w")
            tar_count += 1

        base_name = os.path.splitext(wav_file)[0]  # e.g., sample_0
        txt_file = base_name + ".txt"

        # --- A. 读取各模态数据 ---

        # 1. 音频 (Audio)
        wav_path = os.path.join(SOURCE_DIR, wav_file)
        with open(wav_path, "rb") as f:
            audio_data = f.read()

        # 2. 文本 (Text)
        txt_path = os.path.join(SOURCE_DIR, txt_file)
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                text_content = f.read().strip()
        else:
            text_content = "unknown sound"

        # 3. 图像 (Image - 核心创新点)
        # 实时生成频谱图
        try:
            image_data = create_spectrogram_image(wav_path)
        except Exception as e:
            print(f"生成频谱图失败 {wav_file}: {e}")
            continue

        # 4. 元数据 (JSON) - 适配 MoDE 格式
        # MoDE 需要 'caption' 字段
        metadata = {
            "caption": text_content,
            "key": base_name,
            "audio_path": wav_file
        }
        json_bytes = json.dumps(metadata).encode('utf-8')

        # --- B. 写入 Tar 包 ---

        # 写入 .wav
        info = tarfile.TarInfo(name=f"{base_name}.wav")
        info.size = len(audio_data)
        tar_handle.addfile(info, io.BytesIO(audio_data))

        # 写入 .jpg (频谱图)
        info = tarfile.TarInfo(name=f"{base_name}.jpg")
        info.size = len(image_data)
        tar_handle.addfile(info, io.BytesIO(image_data))

        # 写入 .json
        info = tarfile.TarInfo(name=f"{base_name}.json")
        info.size = len(json_bytes)
        tar_handle.addfile(info, io.BytesIO(json_bytes))

        # 写入 .txt (可选，方便查看)
        info = tarfile.TarInfo(name=f"{base_name}.txt")
        info.size = len(text_content.encode('utf-8'))
        tar_handle.addfile(info, io.BytesIO(text_content.encode('utf-8')))

    # 收尾
    if tar_handle:
        tar_handle.close()

    print("-" * 50)
    print(f"打包完成，共生成 {tar_count} 个 .tar 文件。")
    print(f"输出位置: {DEST_DIR}")
    print("下一步：修改 feature extraction 代码，读取这些三模态数据！")


if __name__ == "__main__":
    pack_data()
