# 350M Algebraic Transformer Hyperparameter Sweep Report (10 Arms)

**Date Completed:** 2026-10-29 11:11:21 IST  
**Cluster Hardware:** Cloud TPU v4-32 (16 TPUs across 4 workers: `w0`, `w1`, `w2`, `w3`)  
**Model Architecture:** 350M Pure Algebraic Transformer (`ffn=alu_glu`, `beta=0.5`, `pos=cayley_geo`, `attn=asinh`, `sink=learned`, `head=alg_ce`)  
**Active Parameters:** 332,453,248 (`d_model=1024`, `n_layers=24`, `n_heads=16`, `d_ff=2688`, `vocab_size=32768`)  
**Acceleration Kernels:** Asinh FlashAttention Pallas + Alg-CE Fused Head Pallas  
**Training Budget:** 600M tokens per arm (573 steps, batch size 512, micro 2, seq length 2,048)  

## Selected Winning Hyperparameters

| Optimizer | Winning Arm | Base LR | Weight Decay | Muon LR | Muon WD | C-Eps | Held-Out NLL | Held-Out PPL |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **muon_adamw** | `alg_350m_muon_adamw_canonical_s42` | 0.002 | 0.1 | 0.02 | 0.1 | None | **3.0423** | **20.95** |
| **hybrid_muon_cadamw** | `alg_350m_hybrid_muon_cadamw_canonical_s42` | 0.002 | 0.1 | 0.02 | 0.1 | 0.001 | **3.0436** | **20.98** |

---

## Full Sweep Arm Results by Optimizer

### Optimizer: `muon_adamw`

| Arm Name | LR | WD | Muon LR | Muon WD | Final Loss | Held-Out NLL | Perplexity | Runtime |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `alg_350m_muon_adamw_base_s42` | 0.0025 | 0.1 | 0.02 | 0.05 | 2.9605 | **3.0536** | 21.19 | 20.4 min |
| `alg_350m_muon_adamw_low_lr_s42` | 0.0015 | 0.1 | 0.015 | 0.05 | 3.0034 | **3.0889** | 21.95 | 20.4 min |
| `alg_350m_muon_adamw_high_lr_s42` | 0.003 | 0.1 | 0.03 | 0.05 | 2.9518 | **3.0453** | 21.02 | 20.4 min |
| `alg_350m_muon_adamw_low_wd_s42` | 0.0025 | 0.05 | 0.02 | 0.02 | 2.9785 | **3.0703** | 21.55 | 20.4 min |
| `alg_350m_muon_adamw_canonical_s42` | 0.002 | 0.1 | 0.02 | 0.1 | 2.9506 | **3.0423** | 20.95 | 20.4 min |

### Optimizer: `hybrid_muon_cadamw`

| Arm Name | LR | WD | Muon LR | Muon WD | Final Loss | Held-Out NLL | Perplexity | Runtime |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `alg_350m_hybrid_muon_cadamw_base_s42` | 0.0025 | 0.1 | 0.02 | 0.05 | 2.9626 | **3.0561** | 21.24 | 20.4 min |
| `alg_350m_hybrid_muon_cadamw_low_lr_s42` | 0.0015 | 0.1 | 0.015 | 0.05 | 2.9994 | **3.0853** | 21.87 | 20.4 min |
| `alg_350m_hybrid_muon_cadamw_high_lr_s42` | 0.003 | 0.1 | 0.03 | 0.05 | 2.9565 | **3.0490** | 21.09 | 20.4 min |
| `alg_350m_hybrid_muon_cadamw_low_wd_s42` | 0.0025 | 0.05 | 0.02 | 0.02 | 2.9796 | **3.0712** | 21.57 | 20.4 min |
| `alg_350m_hybrid_muon_cadamw_canonical_s42` | 0.002 | 0.1 | 0.02 | 0.1 | 2.9533 | **3.0436** | 20.98 | 20.4 min |

