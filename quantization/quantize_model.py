import modal
import subprocess
import os

# Use the same build tools that worked for inference
image = (
    modal.Image.debian_slim()
    .apt_install("git", "build-essential", "cmake", "pkg-config")
)

vol = modal.Volume.from_name("llama31-mlp-only")
app = modal.App("llama-quantizer")

@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=3600, # 1 hour timeout
    cpu=8.0,      # High CPU count speeds up quantization significantly
)
def run_quantization():
    INPUT_MODEL = "/data/pruned_model.gguf"
    OUTPUT_MODEL = "/data/pruned_model_q4_k_m.gguf"
    REPO_DIR = "/root/llama-repo"
    
    # 1. Verification
    if not os.path.exists(INPUT_MODEL):
        print(f"❌ Error: Input model {INPUT_MODEL} not found.")
        return

    # 2. Clone the COMPATIBLE fork (ymcki)
    if not os.path.exists(REPO_DIR):
        print("📥 Cloning llama.cpp (ymcki fork)...")
        subprocess.run(["git", "clone", "https://github.com/ymcki/llama.cpp-b4139", REPO_DIR], check=True)
    
    # 3. Build the quantization tool
    print("🔨 Compiling llama-quantize...")
    # Clean build to ensure no mismatched objects
    subprocess.run(["make", "clean"], cwd=REPO_DIR, check=False) 
    subprocess.run(["make", "llama-quantize", "-j"], cwd=REPO_DIR, check=True)

    # 4. Run Quantization
    # Q4_K_M is the best balance of speed/size/quality for Llama 3 models
    print(f"\n📦 Quantizing model to Q4_K_M...")
    print(f"   Input:  {INPUT_MODEL}")
    print(f"   Output: {OUTPUT_MODEL}")
    print("="*50)
    
    cmd = [
        f"{REPO_DIR}/llama-quantize",
        INPUT_MODEL,
        OUTPUT_MODEL,
        "Q4_K_M"
    ]
    
    # Stream output so you can see the progress bar
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True
    )
    
    for line in process.stdout:
        print(line, end="")
        
    process.wait()
    
    if process.returncode == 0:
        print(f"\n\n✅ Quantization Complete!")
        
        # Show size comparison
        old_size = os.path.getsize(INPUT_MODEL) / (1024**3)
        new_size = os.path.getsize(OUTPUT_MODEL) / (1024**3)
        print(f"   Original Size:  {old_size:.2f} GB")
        print(f"   Quantized Size: {new_size:.2f} GB")
        print(f"   Space Saved:    {old_size - new_size:.2f} GB")
        
        # Commit changes to volume so you can download it
        vol.commit()
    else:
        print(f"\n\n❌ Quantization Failed (Code: {process.returncode})")

@app.local_entrypoint()
def main():

    run_quantization.remote()
