# 125M Standard Transformer Hyperparameter Sweep Report (600M Tokens/Arm)

**Date Completed:** 2026-10-28 23:25:32 IST  
**Cluster Hardware:** Cloud TPU v4-32 (16 TPUs across 4 workers: `w0`, `w1`, `w2`, `w3`)  
**Model Architecture:** 125M Standard LLaMA-style Transformer (`ffn=swiglu`, `pos=rope`, `attn=softmax`, `head=ce`)  
**Acceleration Kernels:** Softmax FlashAttention Pallas + Cross-Entropy Fused Head Pallas  
**Training Budget:** 600M tokens per arm (573 steps, batch size 512, seq length 2,048)  
**Hyperparameter Range:** Grounded in standard 125M literature (GPT-3 / Chinchilla 6e-4, LLaMA 8e-4, Pythia 1e-3)  

## Selected Winning Hyperparameters for Main Runs

| Optimizer | Winning Arm | Base LR | Weight Decay | Muon LR | Muon WD | C-Eps | Held-Out NLL | Held-Out PPL |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **muon_adamw** | `std_125m_muon_adamw_base_s42` | 0.002 | 0.1 | 0.02 | 0.1 | None | **3.1827** | **24.11** |
| **cadamw** | `std_125m_cadamw_lr_1e3_s42` | 0.001 | 0.1 | None | None | 0.001 | **3.7081** | **40.78** |
| **hybrid_muon_cadamw** | `std_125m_hybrid_muon_cadamw_base_s42` | 0.002 | 0.1 | 0.02 | 0.1 | 0.001 | **3.1784** | **24.01** |

---

## Full Sweep Arm Results by Optimizer

### Optimizer: `muon_adamw`

| Arm Name | LR | WD | Muon LR | Muon WD | Final Loss | Held-Out NLL | Perplexity | Runtime |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `std_125m_muon_adamw_base_s42` | 0.002 | 0.1 | 0.02 | 0.1 | 3.1325 | **3.1827** | 24.11 | 6.4 min |
| `std_125m_muon_adamw_canonical_s42` | 0.0006 | 0.1 | 0.02 | 0.1 | 3.2648 | **3.3033** | 27.20 | 6.4 min |
| `std_125m_muon_adamw_canonical_low_muon_s42` | 0.0006 | 0.1 | 0.01 | 0.1 | 3.4024 | **3.4339** | 31.00 | 6.4 min |
| `std_125m_muon_adamw_mid_lr_s42` | 0.001 | 0.1 | 0.02 | 0.1 | 3.1958 | **3.2385** | 25.49 | 6.4 min |
| `std_125m_muon_adamw_low_lr_s42` | 0.001 | 0.1 | 0.01 | 0.1 | 3.3189 | **3.3539** | 28.61 | 6.4 min |

### Optimizer: `cadamw`

| Arm Name | LR | WD | Muon LR | Muon WD | Final Loss | Held-Out NLL | Perplexity | Runtime |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `std_125m_cadamw_lr_6e4_s42` | 0.0006 | 0.1 | - | - | 3.8788 | **3.8932** | 49.07 | 6.2 min |
| `std_125m_cadamw_lr_8e4_s42` | 0.0008 | 0.1 | - | - | 3.7201 | **3.7382** | 42.02 | 6.2 min |
| `std_125m_cadamw_lr_1e3_s42` | 0.001 | 0.1 | - | - | 3.6895 | **3.7081** | 40.78 | 6.2 min |
| `std_125m_cadamw_lr_4e4_s42` | 0.0004 | 0.1 | - | - | 4.1468 | **4.1579** | 63.94 | 6.2 min |
| `std_125m_cadamw_lr_15e4_s42` | 0.0015 | 0.1 | - | - | 3.7090 | **3.7263** | 41.52 | 6.2 min |

### Optimizer: `hybrid_muon_cadamw`

| Arm Name | LR | WD | Muon LR | Muon WD | Final Loss | Held-Out NLL | Perplexity | Runtime |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `std_125m_hybrid_muon_cadamw_canonical_s42` | 0.0006 | 0.1 | 0.02 | 0.1 | 3.2549 | **3.2945** | 26.96 | 6.3 min |
| `std_125m_hybrid_muon_cadamw_canonical_low_muon_s42` | 0.0006 | 0.1 | 0.01 | 0.1 | 3.3862 | **3.4180** | 30.51 | 6.3 min |
| `std_125m_hybrid_muon_cadamw_mid_lr_s42` | 0.001 | 0.1 | 0.02 | 0.1 | 3.1886 | **3.2322** | 25.33 | 6.4 min |
| `std_125m_hybrid_muon_cadamw_low_lr_s42` | 0.001 | 0.1 | 0.01 | 0.1 | 3.3093 | **3.3451** | 28.36 | 6.4 min |
| `std_125m_hybrid_muon_cadamw_base_s42` | 0.002 | 0.1 | 0.02 | 0.1 | 3.1289 | **3.1784** | 24.01 | 6.4 min |

