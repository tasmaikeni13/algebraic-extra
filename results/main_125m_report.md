# 125M Main Training Runs Report (2.5B Tokens, 3 Seeds per Optimizer)

**Date Completed:** 2026-10-28 16:25:02 IST  
**Cluster Hardware:** Cloud TPU v4-32 (16 TPUs across 4 workers: `w0`, `w1`, `w2`, `w3`)  
**Model Architecture:** 125M Algebraic Transformer (`vocab_size=32768`, `d_model=768`, `n_layers=12`, `n_heads=12`, `d_ff=2048`, `seq_len=2048`)  
**Kernel Acceleration:** FlashAttention Pallas (`flash_block=512`) + Algebraic Fused Cross-Entropy Head (`head_block_n=128`, `head_block_v=512`)  
**Training Budget:** 2.5 Billion Tokens per run (2,385 steps, global batch size 512, micro-batch 4 per TPU)  
**Evaluation Setup:** Held-out FineWeb-Edu validation split (8.0M tokens evaluated every 250 steps)  

---

## Executive Summary & Comparison

| Optimizer | Seeds Tested | Held-Out NLL (nats) | Held-Out Perplexity | Final Train Loss | Throughput | Total Time / Run |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Hybrid Muon + C-AdamW** | `42, 43, 44` | **2.7869 ± 0.0012** | **16.23 ± 0.02** | 2.7403 ± 0.0078 | 1,554,056 tok/s | 27.53 min |
| **Muon + AdamW** | `42, 43, 44` | **2.7887 ± 0.0010** | **16.26 ± 0.02** | 2.7405 ± 0.0071 | 1,554,492 tok/s | 27.54 min |
| **Cautious AdamW (C-AdamW)** | `42, 43, 44` | **2.8860 ± 0.0047** | **17.92 ± 0.08** | 2.8395 ± 0.0115 | 1,575,436 tok/s | 27.05 min |

> [!NOTE]
> **Key Takeaway:** **Hybrid Muon + C-AdamW** achieved the best overall validation performance across all 3 random seeds (mean NLL **2.7869** / PPL **16.23**), closely followed by **Muon + AdamW** (mean NLL **2.7887** / PPL **16.26**). Both Muon-based optimizers significantly outperformed standard **C-AdamW** (mean NLL **2.8860** / PPL **17.92**), delivering an **~10% relative reduction in held-out perplexity** (16.23 vs 17.92) for a negligible throughput cost (<1.4% overhead from Newton-Schulz iterations).

---

## Detailed Run Metrics by Optimizer

### **Hybrid Muon + C-AdamW**

| Run Name | Seed | Held-Out NLL | Perplexity | Final Train Loss | Compile Time | Train Time | Throughput |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `main_125m_hybrid_muon_cadamw_s42` | `42` | 2.7856 | 16.21 | 2.7339 | 22.1s | 27.19 min | 1,554,018 tok/s |
| `main_125m_hybrid_muon_cadamw_s43` | `43` | 2.7874 | 16.24 | 2.7379 | 21.2s | 27.18 min | 1,553,719 tok/s |
| `main_125m_hybrid_muon_cadamw_s44` | `44` | 2.7878 | 16.25 | 2.7491 | 20.7s | 27.16 min | 1,554,430 tok/s |

### **Muon + AdamW**

| Run Name | Seed | Held-Out NLL | Perplexity | Final Train Loss | Compile Time | Train Time | Throughput |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `main_125m_muon_adamw_s42` | `42` | 2.7878 | 16.25 | 2.7353 | 22.3s | 27.19 min | 1,554,444 tok/s |
| `main_125m_muon_adamw_s43` | `43` | 2.7884 | 16.26 | 2.7375 | 21.6s | 27.17 min | 1,554,663 tok/s |
| `main_125m_muon_adamw_s44` | `44` | 2.7898 | 16.28 | 2.7486 | 21.2s | 27.17 min | 1,554,370 tok/s |

### **Cautious AdamW (C-AdamW)**

| Run Name | Seed | Held-Out NLL | Perplexity | Final Train Loss | Compile Time | Train Time | Throughput |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `main_125m_cadamw_s42` | `42` | 2.8805 | 17.82 | 2.8280 | 17.8s | 26.76 min | 1,575,170 tok/s |
| `main_125m_cadamw_s43` | `43` | 2.8887 | 17.97 | 2.8397 | 17.6s | 26.76 min | 1,575,135 tok/s |
| `main_125m_cadamw_s44` | `44` | 2.8887 | 17.97 | 2.8509 | 17.6s | 26.74 min | 1,576,004 tok/s |

---

## Position-Wise Evaluation NLL (Context Scaling)

Held-out NLL evaluated across sequence positions ($2^1$ to $2^{12}$ = 2 to 4096 tokens):

| Sequence Position | Hybrid Muon + C-AdamW | Muon + AdamW | C-AdamW |
|:---:|:---:|:---:|:---:|
| **2** | 5.4602 | 5.4582 | 5.5571 |
| **4** | 4.8509 | 4.8652 | 4.9169 |
| **8** | 4.4196 | 4.4324 | 4.4481 |
| **16** | 3.9495 | 3.9667 | 3.9899 |
| **32** | 3.5927 | 3.5982 | 3.6423 |
| **64** | 3.2739 | 3.2807 | 3.3526 |
| **128** | 3.0556 | 3.0587 | 3.1438 |
| **256** | 2.9157 | 2.9184 | 3.0089 |
| **512** | 2.8453 | 2.8468 | 2.9426 |
| **1024** | 2.7941 | 2.7958 | 2.8931 |
| **2048** | 2.7599 | 2.7619 | 2.8599 |
| **4096** | 2.7486 | 2.7500 | 2.8492 |

---

## Cluster Health & Storage Consciousness

- **Zero Checkpoint Leaks:** All intermediate `.npz` model checkpoints were automatically purged immediately upon arm / run completion.
- **Worker Disk Headroom:** Verified across all 4 worker nodes post-execution:
  - `w0`: **64 GB available** (34% utilization)
  - `w1`: **71 GB available** (27% utilization)
  - `w2`: **72 GB available** (26% utilization)
  - `w3`: **72 GB available** (27% utilization)
- **Hardware Utilization:** All 16 TPU v4 cores sustained between **1.55M and 1.58M tokens/second** continuously throughout the entire pipeline without a single OOM or communication hang.
