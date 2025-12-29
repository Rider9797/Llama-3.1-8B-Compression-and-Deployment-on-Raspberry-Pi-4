## LLM Compression & Deployment on Raspberry Pi 4 (Prune $\to$ Convert into GGUF $\to$ Quantize $\to$ Deploy)

## Overview
This repository contains the methodology, scripts, and results for compressing Llama 3.1 8B to run efficiently on a Raspberry Pi 4 (8GB). By combining structural pruning, LoRA fine-tuning, and advanced quantization (AWQ/IMatrix) via llama.cpp, we reduced the model's memory footprint by ~80% while maintaining linguistic coherence.




## Abstract

Deploying state-of-the-art Large Language Models (LLMs) on edge devices is hindered by massive RAM requirements. A standard Llama 3.1 8B model requires ~16GB (FP16), making it impossible to run on a Raspberry Pi 4. This project bridges this gap by applying a multi-stage compression pipeline. We successfully reduced the model size from 16GB to 3.99GB, achieving functional inference on Rasberyy Pi 4 using Llama.cpp.


## Methodology and Pipeline

1. Structural Pruning (MLP-Only)
To maintain compatibility with the Grouped-Query Attention (GQA) in Llama 3.1, we targeted only the MLP intermediate dimensions. This prevents head-count mismatches.

Target: Layers 4 through 27 (Preserves embedding and output layers).

Reduction: 25% reduction in MLP width (14,336 → 10,752).

Result: 13.16% total parameter reduction (8.03B → 6.97B) while keeping critical attention mechanisms intact.
