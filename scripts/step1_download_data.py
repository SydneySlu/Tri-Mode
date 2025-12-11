import os
import io
import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio

# --- Configuration ---

# Mirror configuration for regions with restricted access to Hugging Face
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
print("[Config] Hugging Face Mirror enabled: https://hf-mirror.com")

# Relative path for data storage
SAVE_DIR = "./data/raw"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_and_process():
    print(">>> Initializing ESC-50 dataset download...")
    print("    (Using 'ashraq/esc50' parquet version for stability)")

    try:
        # --- 1. Load Dataset ---
        # streaming=False ensures the entire file is downloaded first to avoid network timeouts
        dataset = load_dataset("ashraq/esc50", split="train", streaming=False)
        
        print("[Info] Metadata loaded successfully.")
        print("[Info] Bypassing automatic decoding (torchcodec workaround)...")

        # --- 2. Bypass Decoding ---
        # Critical fix: Set decode=False to retrieve raw bytes.
        # This prevents the 'torchcodec' error on Windows.
        dataset = dataset.cast_column("audio", Audio(decode=False))

        print(f"[Info] Dataset ready. Total samples available: {len(dataset)}")
        print(">>> Extracting first 20 samples to local storage...")

        count = 0
        target_count = 20

        # --- 3. Iterate and Process ---
        for i in range(len(dataset)):
            if count >= target_count: 
                break

            try:
                sample = dataset[i]

                # Extract raw bytes and label
                # Since decode=False, we get a dictionary: {'bytes': b'...', 'path': ...}
                audio_bytes = sample['audio']['bytes']
                label = sample['category']

                # --- 4. Manual Decoding ---
                # Use soundfile + io to read bytes directly from memory
                audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))

                # --- 5. Construct Caption ---
                caption = f"A sound of {label}."

                # Log progress
                print(f"\n[Sample {count}]")
                print(f"  - Category: {label}")
                print(f"  - Shape:    {audio_array.shape}, SR: {sampling_rate}")

                # --- 6. Save Files ---
                # Save Audio (.wav)
                wav_filename = os.path.join(SAVE_DIR, f"sample_{count}.wav")
                sf.write(wav_filename, audio_array, sampling_rate)

                # Save Text (.txt)
                txt_filename = wav_filename.replace('.wav', '.txt')
                with open(txt_filename, 'w', encoding='utf-8') as f:
                    f.write(caption)

                print(f"  - Saved to: {wav_filename}")
                count += 1

            except Exception as inner_e:
                print(f"[Warning] Skipping sample {i} due to error: {inner_e}")
                continue

        print("-" * 60)
        print(">>> Phase 1: Data Preparation Complete.")
        print(f"    Raw data saved in: {SAVE_DIR}")
        print("-" * 60)

    except Exception as e:
        print(f"\n[Error] Dataset download failed: {e}")
        print("Tip: Check your internet connection or HF_ENDPOINT settings.")

if __name__ == "__main__":
    download_and_process()
