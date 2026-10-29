# 350M Standard Transformer Hyperparameter Sweep Report (10 Arms)

**Date Completed:** 2026-10-29 14:21:01 IST  
**Cluster Hardware:** Cloud TPU v4-32 (16 TPUs across 4 workers: `w0`, `w1`, `w2`, `w3`)  
**Model Architecture:** 350M Standard LLaMA-style Transformer (`ffn=swiglu`, `pos=rope`, `attn=softmax`, `head=ce`)  
**Active Parameters:** 332,449,792 (`d_model=1024`, `n_layers=24`, `n_heads=16`, `d_ff=2688`, `vocab_size=32768`)  
**Acceleration Kernels:** Softmax FlashAttention Pallas + Cross-Entropy Fused Head Pallas  
**Training Budget:** 600M tokens per arm (573 steps, batch size 512, micro 2, seq length 2,048)  

## Selected Winning Hyperparameters

| Optimizer | Winning Arm | Base LR | Weight Decay | Muon LR | Muon WD | C-Eps | Held-Out NLL | Held-Out PPL |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **muon_adamw** | `std_350m_muon_adamw_base_s42` | 0.0015 | 0.1 | 0.02 | 0.1 | None | **3.0625** | **21.38** |
| **hybrid_muon_cadamw** | `std_350m_hybrid_muon_cadamw_base_s42` | 0.0015 | 0.1 | 0.02 | 0.1 | 0.001 | **3.0613** | **21.36** |

---

## Full Sweep Arm Results by Optimizer

### Optimizer: `muon_adamw`

| Arm Name | LR | WD | Muon LR | Muon WD | Final Loss | Held-Out NLL | Perplexity | Runtime |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `std_350m_muon_adamw_canonical_s42` | 0.0005 | 0.1 | 0.02 | 0.1 | 3.0742 | **3.1396** | 23.10 | 18.3 min |
| `std_350m_muon_adamw_canonical_low_muon_s42` | 0.0005 | 0.1 | 0.01 | 0.1 | 3.1879 | **3.2402** | 25.54 | 18.3 min |
| `std_350m_muon_adamw_mid_lr_s42` | 0.0008 | 0.1 | 0.02 | 0.1 | 3.0300 | **3.0997** | 22.19 | 18.2 min |
| `std_350m_muon_adamw_low_lr_s42` | 0.0008 | 0.1 | 0.01 | 0.1 | 3.1375 | **3.1937** | 24.38 | 18.3 min |
| `std_350m_muon_adamw_base_s42` | 0.0015 | 0.1 | 0.02 | 0.1 | 2.9859 | **3.0625** | 21.38 | 18.3 min |

### Optimizer: `hybrid_muon_cadamw`

| Arm Name | LR | WD | Muon LR | Muon WD | Final Loss | Held-Out NLL | Perplexity | Runtime |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `std_350m_hybrid_muon_cadamw_canonical_s42` | 0.0005 | 0.1 | 0.02 | 0.1 | 3.0676 | **3.1339** | 22.96 | 18.3 min |
| `std_350m_hybrid_muon_cadamw_canonical_low_muon_s42` | 0.0005 | 0.1 | 0.01 | 0.1 | 3.1818 | **3.2354** | 25.42 | 18.3 min |
| `std_350m_hybrid_muon_cadamw_mid_lr_s42` | 0.0008 | 0.1 | 0.02 | 0.1 | 3.0212 | **3.0920** | 22.02 | 18.3 min |
| `std_350m_hybrid_muon_cadamw_low_lr_s42` | 0.0008 | 0.1 | 0.01 | 0.1 | 3.1327 | **3.1898** | 24.28 | 18.3 min |
| `std_350m_hybrid_muon_cadamw_base_s42` | 0.0015 | 0.1 | 0.02 | 0.1 | 2.9845 | **3.0613** | 21.36 | 18.3 min |

