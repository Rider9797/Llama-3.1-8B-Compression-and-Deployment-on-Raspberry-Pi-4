## LLM Compression & Deployment on Raspberry Pi 4 (Prune $\to$ Convert into GGUF $\to$ Quantize $\to$ Deploy)

## Overview
This repository contains the methodology, scripts, and results for compressing Llama 3.1 8B to run efficiently on a Raspberry Pi 4 (8GB). By combining structural pruning, LoRA fine-tuning, and advanced quantization (AWQ/IMatrix) via llama.cpp, we reduced the model's memory footprint by ~80% while maintaining linguistic coherence.




## Abstract

Deploying state-of-the-art Large Language Models (LLMs) on edge devices is hindered by massive RAM requirements. A standard Llama 3.1 8B model requires ~16GB (FP16), making it impossible to run on a Raspberry Pi 4. This project bridges this gap by applying a multi-stage compression pipeline. We successfully reduced the model size from 16GB to 3.99GB, achieving functional inference on Rasberyy Pi 4 using Llama.cpp.


## Methodology and Pipeline

1. Structural Pruning (MLP-Only)
## 🛠 Methodology & Pipeline

The project follows a multi-stage compression pipeline to transform a high-RAM requirement model into an edge-compatible executable.

```mermaid
graph LR
    A[Llama 3.1 8B Base] --> B(Structural Pruning)
    B --> C(LoRA Recovery)
    C --> D(GGUF Adaptation)
    D --> E(IMatrix Quantization)
    E --> F[Raspberry Pi 4 Deployment]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style F fill:#d4f1f9,stroke:#333,stroke-width:2px
