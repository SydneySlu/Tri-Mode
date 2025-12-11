import os
import tarfile
import json
import io
import numpy as np
from tqdm import tqdm

# --- Matplotlib Headless Mode Configuration ---
# Critical for running on servers without a display (GUI)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import librosa
import librosa.display

# --- Configuration (Relative Paths) ---
SOURCE_DIR = "./data/raw"      # Output from step1
DEST_DIR = "./data/shards"     # WebDataset Shards
SAMPLES_PER_TAR = 50           # Number of samples per tar file

os.makedirs(DEST_DIR, exist_ok=True)

def create_spectrogram_image(audio_path):
    """
    [Core Innovation] Converts Audio modality to Visual modality (Mel Spectrogram).
    This enables the alignment of Audio, Text, and synthesized Image.
    """
    # 1. Load Audio
    y, sr = librosa.load(audio_path, sr=16000)

    # 2. Compute Mel Spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # 3. Plot and Save to Memory (Bytes)
    # 2.24 inches * 100 dpi = 224 pixels (Standard CLIP input size)
    plt.figure(figsize=(2.24, 2.24), dpi=100) 
    plt.axis('off') # Hide axes
    # Remove padding to ensure the image contains only data
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) 
    
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')

    buf = io.BytesIO()
    plt.savefig(buf, format='jpg', bbox_inches='tight', pad_inches=0)
    plt.close() # Free memory immediately
    buf.seek(0)
    return buf.read()

def pack_data():
    print(">>> Starting WebDataset Packing Pipeline...")
    print(f"[Config] Source: {SOURCE_DIR}")
    print(f"[Config] Target: {DEST_DIR}")
    print(f"[Config] Batch : {SAMPLES_PER_TAR} samples/shard")

    # 1. Collect all wav files
    if not os.path.exists(SOURCE_DIR):
        print(f"[Error] Source directory not found: {SOURCE_DIR}")
        print("       Please run step1_download.py first.")
        return

    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.wav')]
    files.sort() # Ensure deterministic order

    if len(files) == 0:
        print("[Error] No .wav files found in source directory.")
        return

    print(f"[Info] Found {len(files)} samples. Starting packing process...")

    tar_count = 0
    tar_handle = None

    # Use tqdm for progress tracking
    for idx, wav_file in enumerate(tqdm(files, desc="[Packing]")):
        # Rotate tar file every SAMPLES_PER_TAR
        if idx % SAMPLES_PER_TAR == 0:
            if tar_handle:
                tar_handle.close()
            
            tar_name = os.path.join(DEST_DIR, f"{tar_count:08d}.tar")
            tar_handle = tarfile.open(tar_name, "w")
            tar_count += 1

        base_name = os.path.splitext(wav_file)[0] # e.g., sample_0
        txt_file = base_name + ".txt"

        # --- A. Read Data Modalities ---

        # 1. Audio
        wav_path = os.path.join(SOURCE_DIR, wav_file)
        with open(wav_path, "rb") as f:
            audio_data = f.read()

        # 2. Text (Caption)
        txt_path = os.path.join(SOURCE_DIR, txt_file)
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                text_content = f.read().strip()
        else:
            text_content = "unknown sound"

        # 3. Image (Spectrogram) - Generated on-the-fly
        try:
            image_data = create_spectrogram_image(wav_path)
        except Exception as e:
            print(f"\n[Warning] Failed to generate spectrogram for {wav_file}: {e}")
            continue

        # 4. Metadata (JSON) - Compatible with MoDE/MetaCLIP format
        metadata = {
            "caption": text_content,
            "key": base_name,
            "audio_path": wav_file
        }
        json_bytes = json.dumps(metadata).encode('utf-8')

        # --- B. Write to Tar ---

        # Write .wav
        info = tarfile.TarInfo(name=f"{base_name}.wav")
        info.size = len(audio_data)
        tar_handle.addfile(info, io.BytesIO(audio_data))

        # Write .jpg (Spectrogram)
        info = tarfile.TarInfo(name=f"{base_name}.jpg")
        info.size = len(image_data)
        tar_handle.addfile(info, io.BytesIO(image_data))

        # Write .json (Metadata)
        info = tarfile.TarInfo(name=f"{base_name}.json")
        info.size = len(json_bytes)
        tar_handle.addfile(info, io.BytesIO(json_bytes))

        # Write .txt (Optional, for human inspection)
        info = tarfile.TarInfo(name=f"{base_name}.txt")
        info.size = len(text_content.encode('utf-8'))
        tar_handle.addfile(info, io.BytesIO(text_content.encode('utf-8')))

    # Final cleanup
    if tar_handle:
        tar_handle.close()

    print("-" * 60)
    print(">>> Packing Complete.")
    print(f"    Generated {tar_count} shards.")
    print(f"    Output Directory: {DEST_DIR}")
    print("-" * 60)

if __name__ == "__main__":
    pack_data()
