# Copyright (c) Meta Platforms, Inc. and affiliates
# Modified for Tri-Modal Project

import sys
# Ensure local modules can be imported
sys.path.append("src")
sys.path.append("./")

import os

# --- Configuration: HF Mirror ---
# Mirror for regions with restricted access to Hugging Face
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

# Import model libraries
import open_clip
import laion_clap


def get_args():
    parser = argparse.ArgumentParser()
    # Relative paths for portability
    parser.add_argument("--shard_dir", type=str, default="./data/shards", help="Directory containing WebDataset shards")
    parser.add_argument("--output_dir", type=str, default="./data/features", help="Directory to save extracted features")
    return parser.parse_args()


def load_models(device):
    print("[Info] Loading CLIP model (Vision + Text)...")
    # Using ViT-B-32 with OpenAI weights for best compatibility
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    clip_model.to(device)
    clip_model.eval()

    print("[Info] Loading CLAP model (Audio)...")
    # Load LAION-CLAP (HTSAT-base architecture)
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')

    # --- Load Local Weights ---
    # Changed to relative path. Ensure 'clap_model.pt' is in the project root.
    ckpt_path = "./clap_model.pt"

    if os.path.exists(ckpt_path):
        print(f"[Info] Local CLAP weights found: {ckpt_path}. Loading...")
        clap_model.load_ckpt(ckpt=ckpt_path)
    else:
        print(f"[Error] Checkpoint not found: {ckpt_path}")
        print("       Please download the weight file, rename it to 'clap_model.pt', and place it in the project root.")
        return None, None, None  # Exit early

    clap_model.to(device)
    clap_model.eval()

    return clip_model, preprocess, clap_model


def extract_features(shard_path, clip_model, clip_preprocess, clap_model, device):
    print(f"[Info] Processing shard: {os.path.basename(shard_path)}")

    features_list = []
    keys_list = []

    tar = tarfile.open(shard_path)

    for member in tqdm(tar, desc="Processing"):
        if member.name.endswith(".json"):
            base_name = os.path.splitext(member.name)[0]

            try:
                # 1. Read Metadata (Text)
                f = tar.extractfile(member)
                meta = json.load(f)
                text = meta.get("caption", "")

                # 2. Read Image (Spectrogram)
                img_info = tar.getmember(base_name + ".jpg")
                img_file = tar.extractfile(img_info)
                image = Image.open(img_file).convert("RGB")

                # 3. Read Audio
                wav_info = tar.getmember(base_name + ".wav")
                wav_file = tar.extractfile(wav_info)

                # Read bytes and convert to float32
                raw_audio, sr = sf.read(io.BytesIO(wav_file.read()))
                raw_audio = raw_audio.astype(np.float32)

                # Handle multi-channel audio (convert to mono)
                if len(raw_audio.shape) > 1:
                    raw_audio = raw_audio.mean(axis=1)

                # --- Fix: NumPy -> Tensor Conversion ---
                # Reshape to (1, T)
                audio_input_np = raw_audio.reshape(1, -1)
                # Manually convert to Tensor and move to device (CPU)
                audio_input_tensor = torch.from_numpy(audio_input_np).to(device)

                # --- Feature Extraction ---
                with torch.no_grad():
                    # A. CLIP Inference
                    image_input = clip_preprocess(image).unsqueeze(0).to(device)
                    text_input = open_clip.tokenize([text]).to(device)

                    img_feat = clip_model.encode_image(image_input)
                    txt_feat = clip_model.encode_text(text_input)

                    img_feat = F.normalize(img_feat, dim=-1)
                    txt_feat = F.normalize(txt_feat, dim=-1)

                    # B. CLAP Inference
                    audio_feat = clap_model.get_audio_embedding_from_data(x=audio_input_tensor, use_tensor=True)
                    audio_feat = F.normalize(audio_feat, dim=-1).to(device)

                    # C. Feature Fusion
                    img_feat = img_feat.cpu()
                    txt_feat = txt_feat.cpu()
                    audio_feat = audio_feat.cpu()

                    # Concatenate: [1, 512] + [1, 512] + [1, 512] -> [1, 1536]
                    concat_feat = torch.cat([img_feat, txt_feat, audio_feat], dim=-1)

                    features_list.append(concat_feat)
                    keys_list.append(base_name)

            except Exception as e:
                print(f"\n[Warning] Skipping sample {base_name}: {e}")
                pass

    tar.close()

    if len(features_list) > 0:
        return torch.cat(features_list, dim=0), keys_list
    else:
        return None, None


def main():
    args = get_args()
    device = "cpu"  # Force CPU usage

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Models
    clip_model, clip_preprocess, clap_model = load_models(device)
    if clip_model is None:
        return

    # 2. Iterate Shards
    if not os.path.exists(args.shard_dir):
        print(f"[Error] Shard directory not found: {args.shard_dir}")
        return

    files = [f for f in os.listdir(args.shard_dir) if f.endswith(".tar")]
    files.sort()

    if not files:
        print(f"[Error] No .tar files found in {args.shard_dir}")
        return

    for tar_file in files:
        shard_id = tar_file.split(".")[0]  # e.g., 00000000
        save_path = os.path.join(args.output_dir, f"{shard_id}_feat.pth")

        if os.path.exists(save_path):
            print(f"[Info] File exists, skipping: {save_path}")
            continue

        # 3. Extract Features
        full_path = os.path.join(args.shard_dir, tar_file)
        feats, keys = extract_features(full_path, clip_model, clip_preprocess, clap_model, device)

        # 4. Save Results (MoDE Compatible Format)
        if feats is not None:
            data = {
                'feat': feats,         # [N, 1536] Fused features
                'filekeys': keys,      # List of filenames
                'img_midx': list(range(len(keys))) # Mock indices for compatibility
            }
            torch.save(data, save_path)
            print(f"[Success] Features saved to: {save_path} (Shape: {feats.shape})")
        else:
            print(f"[Warning] Extraction result empty for {tar_file}")


if __name__ == "__main__":
    main()
