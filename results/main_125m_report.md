# 125M Main Training Runs Report (2.5B Tokens, 5 Seeds per Optimizer)

**Date Completed:** 2026-10-28 21:28:00 IST  
**Cluster Hardware:** Cloud TPU v4-32 (16 TPUs across 4 workers: `w0`, `w1`, `w2`, `w3`)  
**Model Architecture:** 125M Algebraic Transformer (`vocab_size=32768`, `d_model=768`, `n_layers=12`, `n_heads=12`, `d_ff=2048`, `seq_len=2048`)  
**Kernel Acceleration:** FlashAttention Pallas (`flash_block=512`) + Algebraic Fused Cross-Entropy Head (`head_block_n=128`, `head_block_v=512`)  
**Training Budget:** 2.5 Billion Tokens per run (2,385 steps, global batch size 512, micro-batch 4 per TPU)  
**Evaluation Setup:** Held-out FineWeb-Edu validation split (8.0M tokens evaluated every 250 steps)  

---

## Executive Summary & Comparison

| Optimizer | Seeds Tested | Held-Out NLL (nats) | Held-Out Perplexity | Final Train Loss | Throughput | Total Time / Run |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Hybrid Muon + C-AdamW** | `42, 43, 44, 45, 46` | **2.7864 ± 0.0012** | **16.22 ± 0.02** | 2.7412 ± 0.0069 | 1,553,922 tok/s | 27.53 min |
| **Muon + AdamW** | `42, 43, 44, 45, 46` | **2.7881 ± 0.0011** | **16.25 ± 0.02** | 2.7415 ± 0.0067 | 1,554,397 tok/s | 27.54 min |
| **Cautious AdamW (C-AdamW)** | `42, 43, 44, 45, 46` | **2.8868 ± 0.0035** | **17.94 ± 0.06** | 2.8423 ± 0.0097 | 1,575,246 tok/s | 27.06 min |

> [!NOTE]
> **Key Takeaway:** **Hybrid Muon + C-AdamW** achieved the best overall validation performance across all 5 random seeds (mean NLL **2.7864** / PPL **16.22**), closely followed by **Muon + AdamW** (mean NLL **2.7881** / PPL **16.25**). Both Muon-based optimizers significantly outperformed standard **C-AdamW** (mean NLL **2.8868** / PPL **17.94**), delivering an **~10% relative reduction in held-out perplexity** (16.22 vs 17.94) for a negligible throughput cost (<1.4% overhead from Newton-Schulz iterations).

---

## Detailed Run Metrics by Optimizer

### **Hybrid Muon + C-AdamW**

| Run Name | Seed | Held-Out NLL | Perplexity | Final Train Loss | Compile Time | Train Time | Throughput |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `main_125m_hybrid_muon_cadamw_s42` | `42` | 2.7856 | 16.21 | 2.7339 | 22.1s | 27.19 min | 1,554,018 tok/s |
| `main_125m_hybrid_muon_cadamw_s43` | `43` | 2.7874 | 16.24 | 2.7379 | 21.2s | 27.18 min | 1,553,719 tok/s |
| `main_125m_hybrid_muon_cadamw_s44` | `44` | 2.7878 | 16.25 | 2.7491 | 20.7s | 27.16 min | 1,554,430 tok/s |
| `main_125m_hybrid_muon_cadamw_s45` | `45` | 2.7850 | 16.20 | 2.7481 | 20.7s | 27.17 min | 1,553,764 tok/s |
| `main_125m_hybrid_muon_cadamw_s46` | `46` | 2.7862 | 16.22 | 2.7370 | 21.9s | 27.19 min | 1,553,678 tok/s |

### **Muon + AdamW**

| Run Name | Seed | Held-Out NLL | Perplexity | Final Train Loss | Compile Time | Train Time | Throughput |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `main_125m_muon_adamw_s42` | `42` | 2.7878 | 16.25 | 2.7353 | 22.3s | 27.19 min | 1,554,444 tok/s |
| `main_125m_muon_adamw_s43` | `43` | 2.7884 | 16.26 | 2.7375 | 21.6s | 27.17 min | 1,554,663 tok/s |
| `main_125m_muon_adamw_s44` | `44` | 2.7898 | 16.28 | 2.7486 | 21.2s | 27.17 min | 1,554,370 tok/s |
| `main_125m_muon_adamw_s45` | `45` | 2.7868 | 16.23 | 2.7490 | 21.7s | 27.18 min | 1,554,310 tok/s |
| `main_125m_muon_adamw_s46` | `46` | 2.7878 | 16.24 | 2.7371 | 21.5s | 27.18 min | 1,554,199 tok/s |

### **Cautious AdamW (C-AdamW)**

| Run Name | Seed | Held-Out NLL | Perplexity | Final Train Loss | Compile Time | Train Time | Throughput |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `main_125m_cadamw_s42` | `42` | 2.8805 | 17.82 | 2.8280 | 17.8s | 26.76 min | 1,575,170 tok/s |
| `main_125m_cadamw_s43` | `43` | 2.8887 | 17.97 | 2.8397 | 17.6s | 26.76 min | 1,575,135 tok/s |
| `main_125m_cadamw_s44` | `44` | 2.8887 | 17.97 | 2.8509 | 17.6s | 26.74 min | 1,576,004 tok/s |
| `main_125m_cadamw_s45` | `45` | 2.8884 | 17.96 | 2.8517 | 18.6s | 26.78 min | 1,574,719 tok/s |
| `main_125m_cadamw_s46` | `46` | 2.8877 | 17.95 | 2.8412 | 18.6s | 26.77 min | 1,575,204 tok/s |

---

## Position-Wise Evaluation NLL (Context Scaling)

Held-out NLL evaluated across sequence positions ($2^1$ to $2^{12}$ = 2 to 4096 tokens):

| Sequence Position | Hybrid Muon + C-AdamW | Muon + AdamW | C-AdamW |
|:---:|:---:|:---:|:---:|
| **2** | 5.4591 | 5.4652 | 5.5410 |
| **4** | 4.8496 | 4.8696 | 4.9115 |
| **8** | 4.4216 | 4.4340 | 4.4532 |
| **16** | 3.9516 | 3.9683 | 3.9924 |
| **32** | 3.5910 | 3.5964 | 3.6431 |
| **64** | 3.2729 | 3.2802 | 3.3542 |
| **128** | 3.0555 | 3.0583 | 3.1440 |
| **256** | 2.9151 | 2.9178 | 3.0093 |
| **512** | 2.8445 | 2.8460 | 2.9434 |
| **1024** | 2.7937 | 2.7950 | 2.8939 |
| **2048** | 2.7594 | 2.7613 | 2.8608 |
| **4096** | 2.7481 | 2.7495 | 2.8501 |

---

## Cluster Health & Storage Consciousness

- **Zero Checkpoint Leaks:** All intermediate `.npz` model checkpoints were automatically purged immediately upon arm / run completion.
- **Worker Disk Headroom:** Verified across all 4 worker nodes post-execution:
  - `w0`: **64 GB available** (34% utilization)
  - `w1`: **71 GB available** (27% utilization)
  - `w2`: **72 GB available** (26% utilization)
  - `w3`: **72 GB available** (27% utilization)
- **Hardware Utilization:** All 16 TPU v4 cores sustained between **1.55M and 1.58M tokens/second** continuously throughout the entire pipeline without a single OOM or communication hang.
