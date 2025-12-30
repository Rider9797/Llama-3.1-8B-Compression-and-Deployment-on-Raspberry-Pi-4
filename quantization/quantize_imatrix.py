import modal
import subprocess
import os

image = (
    modal.Image.debian_slim()
    .apt_install("git", "build-essential", "cmake", "pkg-config", "wget", "unzip")
)

vol = modal.Volume.from_name("llama31-mlp-only")
app = modal.App("llama-quantizer-q3-smart")

@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=7200, # 2 hours (IMatrix calculation is slow but worth it)
    cpu=8.0,
)
def run_smart_q3():
    # Paths
    INPUT_MODEL = "/data/pruned_model.gguf"       # Your original FP16 model
    IMATRIX_FILE = "/data/importance_matrix.dat"  # The "Smart Map" file
    OUTPUT_MODEL = "/data/pruned_model_q3_k_m_awq.gguf" # Final Result
    REPO_DIR = "/root/llama-repo"
    
    # Calibration Data Config
    DATASET_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
    DATASET_DIR = "/root/data"
    TRAIN_FILE = f"{DATASET_DIR}/wikitext-2-raw/wiki.train.raw"

    # 1. Setup Tools
    if not os.path.exists(REPO_DIR):
        print("📥 Cloning llama.cpp (ymcki fork)...")
        subprocess.run(["git", "clone", "https://github.com/ymcki/llama.cpp-b4139", REPO_DIR], check=True)
    
    print("🔨 Compiling tools (quantize + imatrix)...")
    # We need both tools for this pipeline
    subprocess.run(["make", "llama-quantize", "llama-imatrix", "-j"], cwd=REPO_DIR, check=True)

    # 2. Download Calibration Data (WikiText)
    # The model needs to read this text to learn which neurons are important
    if not os.path.exists(TRAIN_FILE):
        print("\n⬇️ Downloading calibration dataset...")
        os.makedirs(DATASET_DIR, exist_ok=True)
        subprocess.run(["wget", DATASET_URL, "-O", "wikitext.zip"], cwd=DATASET_DIR, check=True)
        subprocess.run(["unzip", "-o", "wikitext.zip"], cwd=DATASET_DIR, check=True)

    # 3. Calculate IMatrix (The "AWQ" Step)
    # If we already calculated it in a previous run, skip this to save 20 mins
    if os.path.exists(IMATRIX_FILE):
        print(f"\n✅ Found existing IMatrix at {IMATRIX_FILE}. Skipping calculation.")
    else:
        print("\n🧠 CALCULATING IMPORTANCE MATRIX...")
        print("   (This runs the model over data to find the most important weights)")
        print("   (This will take 15-20 minutes on CPU)")
        print("="*50)
        
        imatrix_cmd = [
            f"{REPO_DIR}/llama-imatrix",
            "-m", INPUT_MODEL,
            "-f", TRAIN_FILE,
            "-o", IMATRIX_FILE,
            "-c", "512",      # Context length
            "--chunks", "16"  # Process 16 chunks of data
        ]
        subprocess.run(imatrix_cmd, check=True)
        print("✅ IMatrix calculation complete.")

    # 4. Quantize to Q3 using the IMatrix
    print(f"\n📦 Quantizing to Q3_K_M (Smart Mode)...")
    print("="*50)
    
    quant_cmd = [
        f"{REPO_DIR}/llama-quantize",
        "--imatrix", IMATRIX_FILE,  # <--- This is the magic flag
        INPUT_MODEL,
        OUTPUT_MODEL,
        "Q3_K_M"
    ]
    
    # Run and stream output
    process = subprocess.Popen(quant_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    
    if process.returncode == 0:
        print(f"\n\n✅ Q3 Smart Quantization Complete!")
        # Show stats
        old_size = os.path.getsize(INPUT_MODEL) / (1024**3)
        new_size = os.path.getsize(OUTPUT_MODEL) / (1024**3)
        print(f"   Original (FP16): {old_size:.2f} GB")
        print(f"   Final (Q3 Smart):{new_size:.2f} GB")
        vol.commit()
    else:
        print(f"❌ Failed (Code {process.returncode})")

@app.local_entrypoint()
def main():
    run_smart_q3.remote()