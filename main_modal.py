import modal
import subprocess
import os
import shutil

# --- 1. Define Image (Added Build Tools) ---
image = (
    modal.Image.debian_slim()
    .apt_install(
        "git", "wget", "unzip",
        "cmake", "pkg-config", "build-essential"  # <--- ADDED: Required to compile gguf/sentencepiece
    )
    .pip_install(
        "torch",
        "numpy",
        "sentencepiece",
        "gdown",
        "transformers",
        "accelerate"
    )
    # Add your modified script (the one with the MLP Fix)
    .add_local_file("convert_hf_to_gguf.py", "/root/convert_hf_to_gguf.py")
)

vol = modal.Volume.from_name("llama-pruned-storage", create_if_missing=True)
app = modal.App("llama-folder-converter")

@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=7200,
    gpu="T4",
)
def run_folder_conversion():
    import torch
    import transformers
    
    # --- CONFIGURATION ---
    DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/11_9t7yS6K-epiF8AnVrf0YK7cZQGZZjP"
    
    RAW_DIR = "/data/raw_download"
    CLEAN_DIR = "/data/clean_model"
    GGUF_FILE = "/data/pruned_model.gguf"
    
    # We clone into a specific folder to keep things clean
    REPO_DIR = "/root/llama-repo"
    
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(CLEAN_DIR, exist_ok=True)

    # --- 1. DOWNLOAD FOLDER ---
    if not os.path.exists(f"{RAW_DIR}/pytorch_model.bin"):
        print(f"⬇️ Downloading Folder from Google Drive...")
        try:
            subprocess.run(
                ["gdown", DRIVE_FOLDER_URL, "-O", RAW_DIR, "--folder"], 
                check=True
            )
            print("✅ Download complete!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Download failed. Error: {e}")
            raise
    else:
        print("✅ Raw files found in Volume. Skipping download.")

    # --- 2. CLEAN CHECKPOINT ---
    if not os.path.exists(f"{CLEAN_DIR}/config.json"):
        print("🧹 Cleaning LLM-Pruner checkpoint...")
        checkpoint_path = f"{RAW_DIR}/pytorch_model.bin"
        
        try:
            print(f"   Loading custom checkpoint (Unpickling)...")
            
            # FIX 1: weights_only=False allows loading the custom object
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            
            if 'model' in ckpt:
                print("   Found 'model' key. Extracting...")
                model_obj = ckpt['model']
                print(f"   Saving standard HF format to {CLEAN_DIR}...")
                model_obj.save_pretrained(CLEAN_DIR)
                del model_obj
            else:
                print("❌ 'model' key not found! Keys found:", ckpt.keys())
                raise Exception("Invalid Checkpoint Format")

            if 'tokenizer' in ckpt:
                print("   Found 'tokenizer' key. Saving...")
                ckpt['tokenizer'].save_pretrained(CLEAN_DIR)
            else:
                print("⚠️ No tokenizer found. Attempting manual copy...")
                for f in os.listdir(RAW_DIR):
                    if "token" in f or "vocab" in f:
                        shutil.copy(f"{RAW_DIR}/{f}", f"{CLEAN_DIR}/{f}")

            del ckpt
            print("✅ Cleaning complete.")
            
        except Exception as e:
            print(f"❌ Failed to clean checkpoint: {e}")
            raise
    else:
        print("✅ Cleaned model already exists.")

    # --- 3. PREPARE ENVIRONMENT ---
    if not os.path.exists(REPO_DIR):
        print("📥 Cloning SPECIFIC llama.cpp fork (ymcki)...")
        subprocess.run(["git", "clone", "https://github.com/ymcki/llama.cpp-b4139", REPO_DIR], check=True)
        
        print("🔧 Installing MATCHING gguf library from source...")
        subprocess.run(["pip", "install", f"{REPO_DIR}/gguf-py"], check=True)

    # --- 4. RUN CONVERSION ---
    print("🚀 Running GGUF conversion...")
    
    # We run your LOCALLY mounted script (which has the MLP fix), 
    # but using the environment (gguf lib) we just installed from the repo.
    convert_cmd = [
        "python3", "/root/convert_hf_to_gguf.py",
        CLEAN_DIR,
        "--outfile", GGUF_FILE,
        "--outtype", "f16"
    ]
    
    result = subprocess.run(convert_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Conversion Failed!")
        print(result.stderr)
        raise Exception("Conversion script failed")
    
    print(result.stdout)
    print(f"✅ Conversion complete! File saved at: {GGUF_FILE}")
    vol.commit()

@app.local_entrypoint()
def main():
    run_folder_conversion.remote()