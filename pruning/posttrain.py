# post_train_llama31_mlp.py - Post-training for MLP-only pruned model
import modal

app = modal.App("llm-post-training-mlp")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/horseee/LLM-Pruner.git /root/LLM-Pruner",
        "cd /root/LLM-Pruner && pip install -r requirement.txt"
    )
)

# FIXED: Use the correct volume name
pruned_volume = modal.Volume.from_name("llama31-mlp-only", create_if_missing=False)
tuned_volume = modal.Volume.from_name("llama31-mlp-tuned", create_if_missing=True)

@app.function(
    gpu="A100-80GB",
    memory=81920,
    timeout=18000,  # 5 hours
    image=image,
    volumes={
        "/pruned_models": pruned_volume,
        "/outputs": tuned_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def post_train_llama31_mlp(
    pruned_model_name: str = "llama31_mlp_only",  # FIXED: Match actual name
    num_epochs: int = 2,
    learning_rate: float = 1e-4,
    batch_size: int = 64,
    micro_batch_size: int = 4,
    lora_r: int = 8,
    val_set_size: int = 2000,
    max_train_samples: int = None,
):
    import subprocess
    import os
    import sys
    import re
    
    os.chdir("/root/LLM-Pruner")
    
    pruned_model_path = f"/pruned_models/prune_log/{pruned_model_name}/pytorch_model.bin"
    output_dir = f"/outputs/tune_log/{pruned_model_name}_tuned"
    
    print("="*60)
    print("🔧 POST-TRAINING SETUP (MLP-ONLY MODEL)")
    print("="*60)
    print(f"📥 Model: {pruned_model_path}")
    print(f"📊 Dataset: yahma/alpaca-cleaned")
    print(f"📈 Epochs: {num_epochs}")
    print(f"🔢 Samples: {max_train_samples if max_train_samples else 'All (~50k)'}")
    print(f"📚 LoRA rank: {lora_r}")
    print(f"🎯 Learning rate: {learning_rate}")
    print("="*60)
    print()
    
    # Verify model exists
    if not os.path.exists(pruned_model_path):
        print(f"❌ Model not found: {pruned_model_path}")
        print("\nAvailable in /pruned_models/prune_log:")
        if os.path.exists("/pruned_models/prune_log"):
            for item in os.listdir("/pruned_models/prune_log"):
                print(f"   - {item}")
        return False
    
    print(f"✅ Found model: {os.path.getsize(pruned_model_path) / (1024**3):.2f} GB")
    print()
    
    print("🔧 Applying all necessary patches...")
    
    # ============================================================
    # PATCH 1: Fix post_training.py
    # ============================================================
    print("\n1/2: Patching post_training.py...")
    
    with open("post_training.py", "r") as f:
        content = f.read()
    
    # Fix 1: torch.load (PyTorch 2.6 compatibility)
    if "weights_only=False" not in content:
        content = content.replace(
            "pruned_dict = torch.load(args.prune_model, map_location='cpu')",
            "pruned_dict = torch.load(args.prune_model, map_location='cpu', weights_only=False)"
        )
        print("   ✓ Fixed torch.load (weights_only=False)")
    
    # Fix 2: evaluation_strategy -> eval_strategy
    if 'evaluation_strategy=' in content and 'eval_strategy=' not in content:
        content = content.replace('evaluation_strategy="steps"', 'eval_strategy="steps"')
        print("   ✓ Fixed eval_strategy parameter")
    
    # Fix 3: Disable WandB
    if 'report_to="wandb"' in content:
        content = content.replace('report_to="wandb"', 'report_to="none"')
        print("   ✓ Disabled WandB reporting")
    
    # Fix 4: Limit samples - IMPROVED VERSION
    if max_train_samples:
        # Find the data loading section
        if 'data = load_dataset' in content and f'max_train_samples={max_train_samples}' not in content:
            # Look for the pattern after data loading
            pattern = r'(data = load_dataset\([^)]+\))\s*\n(\s+)(if data_args\.data_path|train_val = data\["train"\])'
            
            replacement = rf'\1\n\2# Limit training samples\n\2if len(data["train"]) > {max_train_samples}:\n\2    data["train"] = data["train"].select(range({max_train_samples}))\n\2    print(f"Limited training to {max_train_samples} samples")\n\2\3'
            
            content_new = re.sub(pattern, replacement, content)
            
            if content_new != content:
                content = content_new
                print(f"   ✓ Limited training samples to {max_train_samples}")
            else:
                # Fallback: simpler approach
                if 'train_val = data["train"].train_test_split' in content:
                    content = content.replace(
                        'train_val = data["train"].train_test_split',
                        f'if len(data["train"]) > {max_train_samples}:\n        data["train"] = data["train"].select(range({max_train_samples}))\n    train_val = data["train"].train_test_split'
                    )
                    print(f"   ✓ Limited training samples to {max_train_samples} (fallback)")
                else:
                    print(f"   ⚠️  Could not apply sample limit")
    
    with open("post_training.py", "w") as f:
        f.write(content)
    
    # ============================================================
    # PATCH 2: Fix LlamaForCausalLM.forward()
    # ============================================================
    print("\n2/2: Patching LlamaForCausalLM.forward()...")
    
    llama_model_path = "LLMPruner/models/hf_llama/modeling_llama.py"
    if os.path.exists(llama_model_path):
        with open(llama_model_path, "r") as f:
            llama_content = f.read()
        
        # Check if already patched
        if '**kwargs' in llama_content and 'def forward(' in llama_content:
            print("   ✓ Already patched, skipping")
        else:
            # Pattern 1: Most common case
            old_pattern = "return_dict: Optional[bool] = None,\n    ) -> Union[Tuple, CausalLMOutputWithPast]:"
            new_pattern = "return_dict: Optional[bool] = None,\n        **kwargs,\n    ) -> Union[Tuple, CausalLMOutputWithPast]:"
            
            if old_pattern in llama_content:
                llama_content = llama_content.replace(old_pattern, new_pattern)
                print("   ✓ Added **kwargs to forward() [Pattern 1]")
            else:
                # Pattern 2: Alternative spacing
                llama_content = llama_content.replace(
                    "return_dict: Optional[bool] = None,\n    )",
                    "return_dict: Optional[bool] = None,\n        **kwargs,\n    )"
                )
                print("   ✓ Added **kwargs to forward() [Pattern 2]")
            
            with open(llama_model_path, "w") as f:
                f.write(llama_content)
    else:
        print("   ⚠️  modeling_llama.py not found, skipping")
    
    print("\n✅ All patches applied!")
    
    # ============================================================
    # RUN TRAINING
    # ============================================================
    print()
    print("="*60)
    print("🚀 STARTING TRAINING")
    print("="*60)
    print()
    
    cmd = [
        "python", "-u", "post_training.py",
        "--prune_model", pruned_model_path,
        "--data_path", "yahma/alpaca-cleaned",
        "--lora_r", str(lora_r),
        "--num_epochs", str(num_epochs),
        "--learning_rate", str(learning_rate),
        "--batch_size", str(batch_size),
        "--micro_batch_size", str(micro_batch_size),
        "--output_dir", output_dir,
        "--val_set_size", str(val_set_size),
    ]
    
    print("Command:", " ".join(cmd))
    print()
    
    # Stream output in real-time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    for line in process.stdout:
        print(line, end='')
        sys.stdout.flush()
    
    process.wait()
    returncode = process.returncode
    
    # ============================================================
    # VERIFY AND SAVE
    # ============================================================
    print()
    print("="*60)
    
    if returncode == 0 and os.path.exists(output_dir):
        print("✅ TRAINING COMPLETED!")
        print("="*60)
        
        import glob
        checkpoints = sorted(glob.glob(f"{output_dir}/checkpoint-*"))
        if checkpoints:
            print(f"\n📋 Generated {len(checkpoints)} checkpoints:")
            for ckpt in checkpoints[-3:]:
                print(f"   - {os.path.basename(ckpt)}")
            
            # Show final checkpoint info
            final_ckpt = checkpoints[-1]
            print(f"\n📦 Final checkpoint: {os.path.basename(final_ckpt)}")
            if os.path.exists(f"{final_ckpt}/adapter_model.bin"):
                size_mb = os.path.getsize(f"{final_ckpt}/adapter_model.bin") / (1024**2)
                print(f"   LoRA weights: {size_mb:.2f} MB")
        
        # List output structure
        print("\n📁 Output structure:")
        for root, dirs, files in os.walk(output_dir):
            level = root.replace(output_dir, '').count(os.sep)
            indent = '  ' * level
            print(f'{indent}{os.path.basename(root)}/')
            sub_indent = '  ' * (level + 1)
            for file in files[:5]:
                size = os.path.getsize(os.path.join(root, file)) / (1024**2)
                print(f'{sub_indent}{file} ({size:.2f} MB)')
            if len(files) > 5:
                print(f'{sub_indent}... and {len(files) - 5} more files')
        
        tuned_volume.commit()
        print("\n✅ Saved to Modal volume 'llama31-mlp-tuned'!")
        
        print("\n📊 Training Summary:")
        print(f"   - Pruned model: {pruned_model_name}")
        print(f"   - Training samples: {max_train_samples or '~50k'}")
        print(f"   - Epochs completed: {num_epochs}")
        print(f"   - Checkpoints saved: {len(checkpoints) if checkpoints else 0}")
        
    else:
        print(f"❌ TRAINING FAILED (exit code: {returncode})")
        print("="*60)
    
    return returncode == 0

@app.local_entrypoint()
def main(
    pruned_model_name: str = "llama31_mlp_only",  # FIXED: Match pruning output
    num_epochs: int = 2,
    learning_rate: float = 1e-4,
    batch_size: int = 64,
    micro_batch_size: int = 4,
    lora_r: int = 8,
    max_train_samples: int = None,
):
    print("="*60)
    print("🚀 LLAMA 3.1 MLP-ONLY POST-TRAINING")
    print("="*60)
    print(f"\n📦 Model: {pruned_model_name}")
    print(f"📊 Dataset: yahma/alpaca-cleaned")
    print(f"📈 Epochs: {num_epochs}")
    print(f"🔢 Samples: {max_train_samples or 'All (~50k)'}")
    print(f"🎯 Learning rate: {learning_rate}")
    print(f"📚 LoRA rank: {lora_r}")
    print("\n⚙️  Hardware:")
    print("   - GPU: A100 80GB")
    print("   - RAM: 80GB")
    print("\n✅ Benefits of MLP-only training:")
    print("   - No attention dimension mismatches")
    print("   - Stable LoRA adapter training")
    print("   - Better convergence expected")
    print("\n⏱️  Expected: 2-3 hours")
    print("💰 Cost: ~$6-8")
    print("="*60)
    print()
    
    success = post_train_llama31_mlp.remote(
        pruned_model_name=pruned_model_name,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        lora_r=lora_r,
        max_train_samples=max_train_samples,
    )
    
    print("\n" + "="*60)
    if success:
        print("✅ SUCCESS!")
        print("="*60)
        print(f"\n📥 Download with:")
        print(f"   modal volume get llama31-mlp-tuned tune_log/{pruned_model_name}_tuned ./tuned_model")
        print("\n📝 Next steps:")
        print("   1. Download the tuned model")
        print("   2. Evaluate on benchmarks (WikiText-2, PTB)")
        print("   3. Test generation quality")
        print("   4. Proceed to quantization (INT8/INT4)")
        print("   5. Convert to GGUF format")
        print("   6. Deploy on Raspberry Pi")
    else:
        print("❌ FAILED")
        print("="*60)
        print("\nCheck the logs above for errors")