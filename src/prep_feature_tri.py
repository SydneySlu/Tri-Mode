# Copyright (c) Meta Platforms, Inc. and affiliates
# Modified for Tri-Modal Project

import sys

sys.path.append("src")
sys.path.append("./")
import os

# 强制使用镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch
import torch.nn.functional as F
import tarfile
import io
import json
import numpy as np
import soundfile as sf
from PIL import Image
from tqdm import tqdm
import argparse

# 引入两个大模型库
import open_clip
import laion_clap


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_dir", type=str, default="D:/TriMode_Data/shards", help="tar包所在目录")
    parser.add_argument("--output_dir", type=str, default="D:/TriMode_Data/features", help="特征输出目录")
    return parser.parse_args()


def load_models(device):
    print("正在加载 CLIP 模型 (用于处理频谱图和文本)...")
    # 使用 ViT-B-32，速度快效果好
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    clip_model.to(device)
    clip_model.eval()

    print("正在加载 CLAP 模型 (用于处理音频)...")
    # 加载 LAION-CLAP (专门对齐音频和文本)
    # amodel='HTSAT-base' 是音频编码器配置
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')

    # --- 加载本地权重 ---
    ckpt_path = "D:/MetaCLIP-main/clap_model.pt"

    if os.path.exists(ckpt_path):
        print(f"检测到本地 CLAP 权重: {ckpt_path}，正在加载...")
        clap_model.load_ckpt(ckpt=ckpt_path)
    else:
        print(f"错误：找不到文件 {ckpt_path}")
        print("请手动下载权重并放在该位置！")
        return None, None, None  # 提前退出

    clap_model.to(device)
    clap_model.eval()

    return clip_model, preprocess, clap_model


def extract_features(shard_path, clip_model, clip_preprocess, clap_model, device):
    print(f"正在处理: {os.path.basename(shard_path)}")

    features_list = []
    keys_list = []

    tar = tarfile.open(shard_path)

    for member in tqdm(tar, desc="Extracting"):
        if member.name.endswith(".json"):
            base_name = os.path.splitext(member.name)[0]

            try:
                # 1. 读取元数据
                f = tar.extractfile(member)
                meta = json.load(f)
                text = meta.get("caption", "")

                # 2. 读取图像
                img_info = tar.getmember(base_name + ".jpg")
                img_file = tar.extractfile(img_info)
                image = Image.open(img_file).convert("RGB")

                # 3. 读取音频
                wav_info = tar.getmember(base_name + ".wav")
                wav_file = tar.extractfile(wav_info)

                # 读取并转 float32
                raw_audio, sr = sf.read(io.BytesIO(wav_file.read()))
                raw_audio = raw_audio.astype(np.float32)

                # 处理双声道
                if len(raw_audio.shape) > 1:
                    raw_audio = raw_audio.mean(axis=1)

                # --- 修正：NumPy -> Tensor ---
                # 重塑为 (1, T)
                audio_input_np = raw_audio.reshape(1, -1)
                # 手动转为 Tensor，并放到正确的设备(CPU)上
                audio_input_tensor = torch.from_numpy(audio_input_np).to(device)

                # --- 特征提取 ---
                with torch.no_grad():
                    # A. CLIP
                    image_input = clip_preprocess(image).unsqueeze(0).to(device)
                    text_input = open_clip.tokenize([text]).to(device)

                    img_feat = clip_model.encode_image(image_input)
                    txt_feat = clip_model.encode_text(text_input)

                    img_feat = F.normalize(img_feat, dim=-1)
                    txt_feat = F.normalize(txt_feat, dim=-1)

                    # B. CLAP (现在输入是 Tensor 了！)
                    audio_feat = clap_model.get_audio_embedding_from_data(x=audio_input_tensor, use_tensor=True)
                    audio_feat = F.normalize(audio_feat, dim=-1).to(device)

                    # C. 融合
                    img_feat = img_feat.cpu()
                    txt_feat = txt_feat.cpu()
                    audio_feat = audio_feat.cpu()

                    # [1, 512] + [1, 512] + [1, 512] -> [1, 1536]
                    concat_feat = torch.cat([img_feat, txt_feat, audio_feat], dim=-1)

                    features_list.append(concat_feat)
                    keys_list.append(base_name)

            except Exception as e:
                print(f"\n跳过样本 {base_name}: {e}")
                # import traceback
                # traceback.print_exc()
                pass

    tar.close()

    if len(features_list) > 0:
        return torch.cat(features_list, dim=0), keys_list
    else:
        return None, None


def main():
    args = get_args()
    device = "cpu"  # 强制 CPU，除非您有显卡

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载模型
    clip_model, clip_preprocess, clap_model = load_models(device)

    # 2. 遍历 shard 目录
    files = [f for f in os.listdir(args.shard_dir) if f.endswith(".tar")]
    files.sort()

    if not files:
        print(f"未在 {args.shard_dir} 找到 .tar 文件！")
        return

    for tar_file in files:
        shard_id = tar_file.split(".")[0]  # 00000000
        save_path = os.path.join(args.output_dir, f"{shard_id}_feat.pth")

        if os.path.exists(save_path):
            print(f"已存在，跳过: {save_path}")
            continue

        # 3. 提取特征
        full_path = os.path.join(args.shard_dir, tar_file)
        feats, keys = extract_features(full_path, clip_model, clip_preprocess, clap_model, device)

        # 4. 保存结果 (适配 MoDE 格式)
        if feats is not None:
            data = {
                'feat': feats,  # [N, 1536] 的融合特征
                'filekeys': keys,  # 文件名列表
                'img_midx': list(range(len(keys)))  # 假索引，适配旧代码
            }
            torch.save(data, save_path)
            print(f"成功保存特征: {save_path} (形状: {feats.shape})")
        else:
            print(f"{tar_file} 提取结果为空")


if __name__ == "__main__":
    main()