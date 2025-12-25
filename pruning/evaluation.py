# eval_perplexity.py - Simple perplexity evaluation
import modal

app = modal.App("llama31-perplexity")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/horseee/LLM-Pruner.git /root/LLM-Pruner",
        "cd /root/LLM-Pruner && pip install -r requirement.txt",
    )
)

pruned_volume = modal.Volume.from_name("llama31-mlp-only", create_if_missing=False)
tuned_volume = modal.Volume.from_name("llama31-mlp-tuned", create_if_missing=False)

@app.function(
    gpu="A100-40GB",
    memory=40960,
    timeout=3600,
    image=image,
    volumes={"/pruned": pruned_volume, "/tuned": tuned_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def eval_ppl(checkpoint: str = "checkpoint-1556"):
    import torch
    import os
    import sys
    import shutil
    from datasets import load_dataset
    from tqdm import tqdm
    
    os.chdir("/root/LLM-Pruner")
    sys.path.insert(0, "/root/LLM-Pruner")
    
    from LLMPruner.peft import PeftModel
    
    # Setup
    base = "/tuned/tune_log/llama31_mlp_only_tuned"
    ckpt_path = f"{base}/{checkpoint}"
    
    if not os.path.exists(f"{ckpt_path}/adapter_model.bin"):
        shutil.copy(f"{base}/adapter_model.bin", f"{ckpt_path}/adapter_model.bin")
    
    print("Loading model...")
    pruned_dict = torch.load(
        "/pruned/prune_log/llama31_mlp_only/pytorch_model.bin",
        map_location='cpu',
        weights_only=False
    )
    tokenizer = pruned_dict['tokenizer']
    model = pruned_dict['model']
    
    model = PeftModel.from_pretrained(model, ckpt_path, torch_dtype=torch.float16)
    model.half().cuda().eval()
    
    print("\n" + "="*60)
    print("PERPLEXITY EVALUATION")
    print("="*60)
    
    # WikiText-2
    print("\n1. WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    
    encodings = tokenizer(text, return_tensors="pt")
    max_length = 2048
    seq_len = encodings.input_ids.size(1)
    
    nlls = []
    prev_end_loc = 0
    
    for begin_loc in tqdm(range(0, min(seq_len, 10000), max_length), desc="WikiText-2"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].cuda()
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
        
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        
        if end_loc >= 10000:
            break
    
    ppl_wikitext = torch.exp(torch.stack(nlls).sum() / prev_end_loc).item()
    
    # PTB
    print("\n2. PTB...")
    try:
        dataset = load_dataset("ptb-text-only/ptb_text_only", split="test")
    except:
        print("   PTB not available, using WikiText validation instead")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    
    text = "\n\n".join([t for t in dataset[list(dataset.features.keys())[0]] if str(t).strip()])
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    
    nlls = []
    prev_end_loc = 0
    
    for begin_loc in tqdm(range(0, min(seq_len, 10000), max_length), desc="PTB"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].cuda()
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
        
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        
        if end_loc >= 10000:
            break
    
    ppl_ptb = torch.exp(torch.stack(nlls).sum() / prev_end_loc).item()
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nWikiText-2 Perplexity: {ppl_wikitext:.2f}")
    print(f"PTB Perplexity:        {ppl_ptb:.2f}")
    
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print("Original Llama 3.1 8B:  ~6-8")
    print("After MLP pruning:      ~9-12")
    print("After post-training:    ~8-10 (expected)")
    print(f"Your model:             {ppl_wikitext:.2f}")
    
    if ppl_wikitext <= 10:
        print("\n✅ EXCELLENT - Comparable to original!")
    elif ppl_wikitext <= 12:
        print("\n✅ GOOD - Acceptable quality")
    else:
        print("\n⚠️ HIGH - May need more training")
    
    return {
        "wikitext2": ppl_wikitext,
        "ptb": ppl_ptb,
    }

@app.local_entrypoint()
def main(checkpoint: str = "checkpoint-1556"):
    print(f"Evaluating {checkpoint}...\n")
    results = eval_ppl.remote(checkpoint=checkpoint)
    print("\n✅ Evaluation complete!")
    print(f"\nWikiText-2: {results['wikitext2']:.2f}")
    print(f"PTB: {results['ptb']:.2f}")