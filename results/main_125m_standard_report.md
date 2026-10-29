# 125M Standard Transformer Main Training Runs Report (2.5B Tokens, 5 Seeds)

**Date Completed:** 2026-10-29 05:52:39 IST  
**Cluster Hardware:** Cloud TPU v4-32 (16 TPUs across 4 workers: `w0`, `w1`, `w2`, `w3`)  
**Model Architecture:** 125M Standard LLaMA-style Transformer (`ffn=swiglu`, `pos=rope`, `attn=softmax`, `head=ce`)  
**Kernel Acceleration:** Softmax FlashAttention Pallas + Cross-Entropy Fused Head Pallas  
**Training Budget:** 2.5 Billion Tokens per run (2,385 steps, global batch size 512, micro-batch 4 per TPU)  
**Evaluation Setup:** Held-out FineWeb-Edu validation split (8.0M tokens evaluated every 250 steps)  

## Executive Summary & Comparison across Optimizers (5 Seeds)

| Optimizer | Seeds Tested | Held-Out NLL (nats) | Held-Out Perplexity | Final Train Loss | Throughput | Total Time / Run |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Muon + AdamW** | `42, 43, 44, 45, 46` | **2.8130 ± 0.0015** | **16.66 ± 0.02** | 2.7688 ± 0.0060 | 1,656,248 tok/s | 25.59 min |
| **Cautious AdamW (C-AdamW)** | `42, 43, 44, 45, 46` | **2.9847 ± 0.0042** | **19.78 ± 0.08** | 2.9425 ± 0.0103 | 1,680,364 tok/s | 25.15 min |
| **Hybrid Muon + C-AdamW** | `42, 43, 44, 45, 46` | **2.8149 ± 0.0018** | **16.69 ± 0.03** | 2.7704 ± 0.0053 | 1,655,655 tok/s | 25.61 min |

---

## Detailed Run Metrics by Optimizer

### **Muon + AdamW**

| Run Name | Seed | Held-Out NLL | Perplexity | Final Train Loss | Compile Time | Train Time | Throughput |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `main_std_125m_muon_adamw_s42` | `42` | 2.8111 | 16.63 | 2.7620 | 13.0s | 25.38 min | 1,656,103 tok/s |
| `main_std_125m_muon_adamw_s43` | `43` | 2.8129 | 16.66 | 2.7645 | 12.4s | 25.37 min | 1,656,526 tok/s |
| `main_std_125m_muon_adamw_s44` | `44` | 2.8133 | 16.66 | 2.7759 | 13.0s | 25.38 min | 1,656,279 tok/s |
| `main_std_125m_muon_adamw_s45` | `45` | 2.8124 | 16.65 | 2.7739 | 12.8s | 25.38 min | 1,656,092 tok/s |
| `main_std_125m_muon_adamw_s46` | `46` | 2.8152 | 16.70 | 2.7676 | 12.6s | 25.38 min | 1,656,238 tok/s |

### **Cautious AdamW (C-AdamW)**

| Run Name | Seed | Held-Out NLL | Perplexity | Final Train Loss | Compile Time | Train Time | Throughput |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `main_std_125m_cadamw_s42` | `42` | 2.9784 | 19.66 | 2.9295 | 10.3s | 24.98 min | 1,680,340 tok/s |
| `main_std_125m_cadamw_s43` | `43` | 2.9866 | 19.82 | 2.9399 | 10.0s | 24.97 min | 1,680,447 tok/s |
| `main_std_125m_cadamw_s44` | `44` | 2.9873 | 19.83 | 2.9540 | 10.3s | 24.98 min | 1,680,325 tok/s |
| `main_std_125m_cadamw_s45` | `45` | 2.9886 | 19.86 | 2.9521 | 10.4s | 24.98 min | 1,680,298 tok/s |
| `main_std_125m_cadamw_s46` | `46` | 2.9825 | 19.74 | 2.9373 | 10.5s | 24.98 min | 1,680,412 tok/s |

### **Hybrid Muon + C-AdamW**

| Run Name | Seed | Held-Out NLL | Perplexity | Final Train Loss | Compile Time | Train Time | Throughput |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `main_std_125m_hybrid_muon_cadamw_s42` | `42` | 2.8144 | 16.68 | 2.7651 | 13.2s | 25.40 min | 1,655,565 tok/s |
| `main_std_125m_hybrid_muon_cadamw_s43` | `43` | 2.8156 | 16.70 | 2.7656 | 12.3s | 25.38 min | 1,655,842 tok/s |
| `main_std_125m_hybrid_muon_cadamw_s44` | `44` | 2.8140 | 16.68 | 2.7770 | 13.4s | 25.40 min | 1,655,548 tok/s |
| `main_std_125m_hybrid_muon_cadamw_s45` | `45` | 2.8130 | 16.66 | 2.7747 | 13.5s | 25.40 min | 1,655,522 tok/s |
| `main_std_125m_hybrid_muon_cadamw_s46` | `46` | 2.8177 | 16.74 | 2.7696 | 12.4s | 25.38 min | 1,655,800 tok/s |

---

## Position-Wise Evaluation NLL (Context Scaling)

Held-out NLL evaluated across sequence positions ($2^1$ to $2^{12}$ = 2 to 4096 tokens):

| Sequence Position | Hybrid Muon + C-AdamW | Muon + AdamW | C-AdamW |
|:---:|:---:|:---:|:---:|
| **2** | 5.8524 | 5.8271 | 5.6234 |
| **4** | 4.9808 | 4.9718 | 5.0308 |
| **8** | 4.4744 | 4.4688 | 4.5552 |
| **16** | 3.9734 | 3.9678 | 4.0815 |
| **32** | 3.6052 | 3.6011 | 3.7391 |
| **64** | 3.2972 | 3.2954 | 3.4440 |
| **128** | 3.0786 | 3.0773 | 3.2352 |
| **256** | 2.9391 | 2.9365 | 3.1024 |
| **512** | 2.8702 | 2.8681 | 3.0376 |
| **1024** | 2.8202 | 2.8186 | 2.9902 |
| **2048** | 2.7877 | 2.7858 | 2.9587 |
| **4096** | 2.7778 | 2.7758 | 2.9495 |

---

## Cluster Health & Storage Consciousness

- **Zero Checkpoint Leaks:** All intermediate `.npz` model checkpoints were automatically purged immediately upon arm / run completion.
- **Worker Disk Headroom:** Verified across all 4 worker nodes post-execution (>64 GB free on each worker).
- **Hardware Utilization:** All 16 TPU v4 cores sustained between **1.65M and 1.68M tokens/second** throughout the standard benchmark without interruption.
