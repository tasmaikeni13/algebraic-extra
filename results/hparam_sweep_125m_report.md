# 125M Hyperparameter Sweep Report (600M Tokens per Arm)

**Generated:** 2026-10-28 12:17:26 IST  
**Cluster Hardware:** Cloud TPU v4-32 (16 TPUs across 4 workers: w0, w1, w2, w3)  
**Model Scale:** 125M Algebraic Transformer (`attn_impl=flash`, `head_impl=fused`)  
**Training Budget:** 600M tokens per arm (573 steps, batch size 512, seq length 2,048)  

## Selected Hyperparameters for Main Runs

| Optimizer | Winning Arm | Learning Rate | Weight Decay | Muon LR | Muon WD | Cautious Eps | Held-Out NLL | Held-Out PPL |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **muon_adamw** | `125m_muon_adamw_high_wd_s42` | 0.003 | 0.1 | 0.02 | 0.1 | 0.001 | **3.1267** | **22.80** |
| **cadamw** | `125m_cadamw_low_lr_s42` | 0.0015 | 0.1 | None | None | 0.001 | **3.6361** | **37.94** |
| **hybrid_muon_cadamw** | `125m_hybrid_muon_cadamw_high_wd_s42` | 0.003 | 0.1 | 0.02 | 0.1 | 0.001 | **3.1286** | **22.84** |

---

## Full Sweep Results by Optimizer

### Optimizer: `muon_adamw`

| Arm Name | LR | WD | Muon LR | Muon WD | C-Eps | Final Loss | Held-Out NLL | Perplexity | Runtime |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `125m_muon_adamw_high_wd_s42` | 0.003 | 0.1 | 0.02 | 0.1 | 0.001 | 3.0657 | **3.1267** | 22.80 | 7.0 min |
| `125m_muon_adamw_base_s42` | 0.003 | 0.1 | 0.02 | 0.05 | 0.001 | 3.0804 | **3.1417** | 23.14 | 6.9 min |
| `125m_muon_adamw_high_lr_s42` | 0.003 | 0.1 | 0.04 | 0.05 | 0.001 | 3.0850 | **3.1444** | 23.21 | 6.9 min |
| `125m_muon_adamw_low_wd_s42` | 0.003 | 0.1 | 0.02 | 0.01 | 0.001 | 3.0976 | **3.1574** | 23.51 | 7.0 min |
| `125m_muon_adamw_low_lr_s42` | 0.0015 | 0.1 | 0.01 | 0.05 | 0.001 | 3.1721 | **3.2236** | 25.12 | 7.0 min |

### Optimizer: `cadamw`

| Arm Name | LR | WD | Muon LR | Muon WD | C-Eps | Final Loss | Held-Out NLL | Perplexity | Runtime |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `125m_cadamw_low_lr_s42` | 0.0015 | 0.1 | None | None | 0.001 | 3.6182 | **3.6361** | 37.94 | 6.8 min |
| `125m_cadamw_base_s42` | 0.003 | 0.1 | None | None | 0.001 | 4.0715 | **4.0863** | 59.52 | 6.8 min |
| `125m_cadamw_low_wd_s42` | 0.003 | 0.05 | None | None | 0.001 | 4.1873 | **4.1989** | 66.61 | 6.8 min |
| `125m_cadamw_high_wd_s42` | 0.003 | 0.2 | None | None | 0.001 | 4.1908 | **4.2024** | 66.85 | 6.8 min |
| `125m_cadamw_high_lr_s42` | 0.0045 | 0.1 | None | None | 0.001 | 4.6256 | **4.6273** | 102.23 | 6.8 min |

### Optimizer: `hybrid_muon_cadamw`

| Arm Name | LR | WD | Muon LR | Muon WD | C-Eps | Final Loss | Held-Out NLL | Perplexity | Runtime |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `125m_hybrid_muon_cadamw_high_wd_s42` | 0.003 | 0.1 | 0.02 | 0.1 | 0.001 | 3.0675 | **3.1286** | 22.84 | 7.0 min |
| `125m_hybrid_muon_cadamw_high_lr_s42` | 0.003 | 0.1 | 0.04 | 0.05 | 0.001 | 3.0833 | **3.1422** | 23.15 | 7.0 min |
| `125m_hybrid_muon_cadamw_base_s42` | 0.003 | 0.1 | 0.02 | 0.05 | 0.001 | 3.0831 | **3.1429** | 23.17 | 7.0 min |
| `125m_hybrid_muon_cadamw_low_wd_s42` | 0.003 | 0.1 | 0.02 | 0.01 | 0.001 | 3.0985 | **3.1580** | 23.52 | 6.9 min |
| `125m_hybrid_muon_cadamw_low_lr_s42` | 0.0015 | 0.1 | 0.01 | 0.05 | 0.001 | 3.1670 | **3.2193** | 25.01 | 7.0 min |

