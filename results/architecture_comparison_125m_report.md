# Architecture Benchmark Comparison: Algebraic vs. Standard Transformer

**Generated:** 2026-10-29 05:51:13 IST  
**Scale:** 125M Active Parameters (`~110.1M` non-embedding weights, tied embeddings)  
**Dataset:** FineWeb-Edu (2.5 Billion Tokens per run)  
**Hardware:** Cloud TPU v4-32 (16 TPUs across 4 workers: `w0`, `w1`, `w2`, `w3`)  
**Seeds Evaluated:** 5 seeds per optimizer (`42, 43, 44, 45, 46`) = 30 full training runs total  

## Head-to-Head Results Table (5 Seeds per Arm)

| Architecture | Optimizer | Held-Out NLL (nats) | Held-Out Perplexity | Final Train Loss | Delta NLL vs Standard | Relative PPL Reduction |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Standard LLaMA** | **Muon + AdamW** | 2.8130 ± 0.0015 | 16.66 ± 0.02 | 2.7688 | baseline | baseline |
| **Algebraic Transformer** | **Muon + AdamW** | **2.7881 ± 0.0011** | **16.25 ± 0.02** | 2.7415 | **-0.0249** | **+2.46%** |
| **Standard LLaMA** | **Cautious AdamW (C-AdamW)** | 2.9847 ± 0.0042 | 19.78 ± 0.08 | 2.9425 | baseline | baseline |
| **Algebraic Transformer** | **Cautious AdamW (C-AdamW)** | **2.8868 ± 0.0035** | **17.94 ± 0.06** | 2.8423 | **-0.0979** | **+9.32%** |
| **Standard LLaMA** | **Hybrid Muon + C-AdamW** | 2.8149 ± 0.0018 | 16.69 ± 0.03 | 2.7704 | baseline | baseline |
| **Algebraic Transformer** | **Hybrid Muon + C-AdamW** | **2.7864 ± 0.0012** | **16.22 ± 0.02** | 2.7412 | **-0.0286** | **+2.81%** |
