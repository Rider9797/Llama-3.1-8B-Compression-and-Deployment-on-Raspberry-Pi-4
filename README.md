# Llama 3.1 Compressiong Pipeline (Prune $\to$ GGUF $\to$ Quantize)

This repository contains a complete, cloud-based pipeline to shrink **Llama 3.1 8B** for efficient inference on edge devices (like Raspberry Pi 5). It utilizes [Modal](https://modal.com) to leverage high-performance cloud GPUs (A100/T4) for the heavy lifting.

The pipeline performs three sequential steps:
1.  **Pruning:** Removes 25% of the model's weights (specifically targeting MLP layers) to lower RAM usage while preserving attention mechanisms.
2.  **Conversion:** Converts the "jagged" pruned model architecture into the standard GGUF format.
3.  **Quantization:** Compresses the model bits (to 4-bit or 3-bit) for final deployment.

## File Overview

| File | Description |
| :--- | :--- |
| `pruning.py` | Patches `LLM-Pruner` and runs MLP-only pruning on an A100 GPU. Saves results to the cloud volume. |
| `conversion.py` | Downloads the pruned checkpoint from the volume, "cleans" the custom architecture, and converts it to FP16 GGUF. |
| `quantize_model.py` | **(Option A)** Performs standard quantization (Fastest setup). |
| `quantize_imatrix.py` | **(Option B)** Performs "Data-Aware" quantization using an Importance Matrix. |
| `convert_hf_to_gguf.py` | A patched version of the official script that handles variable MLP sizes (critical for pruned models). |

## Prerequisites

1.  **Modal Account:**
    * Install the Modal client: `pip install modal`
    * Setup your account: `modal setup`

2.  **Hugging Face Token:**
    * You must create a Secret in Modal named `huggingface-secret`.
    * It should contain your `HF_TOKEN` (required to download Llama 3.1).

3.  **Volume Configuration:**
    * Ensure all scripts reference the same volume name: `llama31-mlp-only`.
    * *Note: If you copied scripts from earlier versions, ensure `vol = modal.Volume.from_name("llama31-mlp-only")` is set in the quantization files.*

## 🚀 Usage Guide

### Step 1: Pruning
Run the pruning job on a cloud A100 GPU. This script automatically applies necessary fixes to the `LLM-Pruner` library.

```bash
modal run pruning.py
```

- Input: meta-llama/Llama-3.1-8B-Instruct
- Action: Prunes 25% of MLP layers.
- Output: Saved to Modal Volume llama31-mlp-only.

### Step 2: Conversion to GGUF Format
Mounts the volume, standardizes the checkpoint structure, and creates a raw FP16 GGUF file.

```bash
modal run conversion.py
```

- Output: /data/pruned_model.gguf (inside the cloud volume).

### Step 3: Quantization
Choose one of the following options based on your target hardware.

#### Option A: Standard Quantization (Fastest)
Best for 4-bit models (Q4_K_M) or if you want results quickly.
Run the pruning job on a cloud A100 GPU. This script automatically applies necessary fixes to the `LLM-Pruner` library.

```bash
modal run quantize_model.py
```

#### Option B: Smart IMatrix Quantization (Recommended for Q3)
Best for 3-bit models (Q3_K_M). It calculates an "Importance Matrix" to identify and protect essential weights, preventing the quality degradation usually associated with heavy compression.

```bash
modal run quantize_imatrix.py
```



