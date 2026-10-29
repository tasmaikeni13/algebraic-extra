# 125M Standard Transformer Main Training Runs Report (2.5B Tokens, 5 Seeds)

**Date Completed:** 2026-10-29 05:51:13 IST  
**Cluster Hardware:** Cloud TPU v4-32 (16 TPUs across 4 workers: `w0`, `w1`, `w2`, `w3`)  
**Model Architecture:** 125M Standard LLaMA-style Transformer (`ffn=swiglu`, `pos=rope`, `attn=softmax`, `head=ce`)  
**Kernel Acceleration:** Softmax FlashAttention Pallas + Cross-Entropy Fused Head Pallas  
**Training Budget:** 2.5 Billion Tokens per run (2,385 steps, global batch size 512, micro-batch 4 per TPU)  

## Executive Summary & Comparison across Optimizers (5 Seeds)

| Optimizer | Seeds Tested | Held-Out NLL (nats) | Held-Out Perplexity | Final Train Loss | Throughput | Total Time / Run |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Muon + AdamW** | `42, 43, 44, 45, 46` | **2.8130 ± 0.0015** | **16.66 ± 0.02** | 2.7688 ± 0.0060 | 1,656,248 tok/s | 25.59 min |
| **Cautious AdamW (C-AdamW)** | `42, 43, 44, 45, 46` | **2.9847 ± 0.0042** | **19.78 ± 0.08** | 2.9425 ± 0.0103 | 1,680,364 tok/s | 25.15 min |
| **Hybrid Muon + C-AdamW** | `42, 43, 44, 45, 46` | **2.8149 ± 0.0018** | **16.69 ± 0.03** | 2.7704 ± 0.0053 | 1,655,655 tok/s | 25.61 min |
