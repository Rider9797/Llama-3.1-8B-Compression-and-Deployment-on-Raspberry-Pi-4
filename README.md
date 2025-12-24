# Pruned Llama-2/3 to GGUF Conversion Pipeline (Modal)

This repository contains a cloud-based pipeline for converting **structurally pruned** Llama models (specifically those with "Variable MLPs" or jagged tensors from tools like LLM-Pruner) into **GGUF format** for use with `llama.cpp`.

It uses [Modal](https://modal.com/) to handle the heavy lifting (downloading, cleaning, converting, and quantizing) on high-performance cloud GPUs/CPUs, solving common issues with custom checkpoints that standard conversion scripts cannot handle.

## 📂 File Overview

* **`modal_main.py`**: The entry point. It connects to Google Drive, downloads the raw custom checkpoint (`pytorch_model.bin`), "cleans" it (unpickles the custom model object and saves it to standard Hugging Face format), and runs the conversion to FP16 GGUF.
* **`quantize_model.py`**: Takes the converted FP16 GGUF from the cloud volume and compresses it to **Q4_K_M** (4-bit quantization) for efficient inference on edge devices (Raspberry Pi, laptops, etc.).
* **`convert_hf_to_gguf.py`**: A **patched version** of the standard `llama.cpp` converter. It includes a critical fix for "Variable MLP" sizes, allowing it to process pruned models where different layers have different hidden sizes.

## Prerequisites

1.  **Modal Account**: Sign up at [modal.com](https://modal.com/).
2.  **Python Environment**:
    ```bash
    pip install modal
    modal setup
    ```
3.  **Google Drive Link**: You need a Google Drive folder containing your pruned `pytorch_model.bin`.
    * *Important:* The folder must be shared as **"Anyone with the link" -> "Viewer"**.

## Usage Guide

### Step 1: Configure & Convert (`modal_main.py`)
Open `modal_main.py` and update the `DRIVE_FOLDER_URL` variable with your specific Google Drive folder link.

Then, run the script to download, clean, and convert the model to FP16:

```bash
modal run modal_main.py
```

### Step 2: Quantize (`quantize_model.py`)
Run the script to quantize the pruned gguf file.
```bash
modal run modal_main.py
```
