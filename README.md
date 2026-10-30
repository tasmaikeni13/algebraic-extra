# Algebraic Transformer (`algebraic-extra`)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hardware: Cloud TPU v4-32](https://img.shields.io/badge/Hardware-Cloud%20TPU%20v4--32-orange.svg)](https://cloud.google.com/tpu)
[![Framework: JAX / Pallas](https://img.shields.io/badge/Framework-JAX%20%7C%20Pallas-blue.svg)](https://github.com/google/jax)

This repository contains supplemental research benchmarks investigating the empirical scaling behavior, hyperparameter sensitivity, and loss surface geometry of the **Algebraic Transformer** versus the **Standard LLaMA-style Transformer** under different modern optimizers (`muon_adamw`, `cadamw`, and `hybrid_muon_cadamw`) at 125M and 350M parameter scales on Cloud TPU v4-32 pods.

> **Primary Research & Core Codebase:**  
> The theoretical foundations, primitive definitions, and authoritative implementation are documented at:  
> **[https://github.com/tasmaikeni13/algebraic-transformer](https://github.com/tasmaikeni13/algebraic-transformer)**

---

## 1. 125M Parameter Benchmarks (5 Random Seeds, 2.5B Tokens Each)

Trained from scratch on FineWeb-Edu (2.5B tokens / 2,385 steps per run, batch size 512, sequence length 2,048 across 16 Cloud TPU v4 chips). Evaluated across 5 independent random seeds (`42, 43, 44, 45, 46`):

| Architecture | Optimizer | Held-Out NLL (nats) | Held-Out Perplexity | Final Train Loss | Delta NLL vs. Standard | Relative PPL Reduction |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Standard LLaMA** | **Muon + AdamW** | 2.8130 ± 0.0015 | 16.66 ± 0.02 | 2.7688 | *baseline* | *baseline* |
| **Algebraic Transformer** | **Muon + AdamW** | **2.7881 ± 0.0011** | **16.25 ± 0.02** | **2.7415** | **-0.0249** | **+2.46%** |
| **Standard LLaMA** | **Cautious AdamW (C-AdamW)** | 2.9847 ± 0.0042 | 19.78 ± 0.08 | 2.9425 | *baseline* | *baseline* |
| **Algebraic Transformer** | **Cautious AdamW (C-AdamW)** | **2.8868 ± 0.0035** | **17.94 ± 0.06** | **2.8423** | **-0.0979** | **+9.32%** |
| **Standard LLaMA** | **Hybrid Muon + C-AdamW** | 2.8149 ± 0.0018 | 16.69 ± 0.03 | 2.7704 | *baseline* | *baseline* |
| **Algebraic Transformer** | **Hybrid Muon + C-AdamW** | **2.7864 ± 0.0012** | **16.22 ± 0.02** | **2.7412** | **-0.0286** | **+2.81%** |

### Context Scaling: Position-Wise NLL ($2^1$ to $2^{12}$ tokens)

Held-out validation NLL at power-of-2 context positions:

| Position | Algebraic (Hybrid) | Standard (Hybrid) | Delta (Hybrid) | Algebraic (C-AdamW) | Standard (C-AdamW) | Delta (C-AdamW) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **2** | **5.4591** | 5.8524 | **-0.3933** | **5.5410** | 5.6234 | **-0.0823** |
| **4** | **4.8496** | 4.9808 | **-0.1313** | **4.9115** | 5.0308 | **-0.1193** |
| **8** | **4.4216** | 4.4744 | **-0.0528** | **4.4532** | 4.5552 | **-0.1020** |
| **16** | **3.9516** | 3.9734 | **-0.0218** | **3.9924** | 4.0815 | **-0.0891** |
| **32** | **3.5910** | 3.6052 | **-0.0143** | **3.6431** | 3.7391 | **-0.0960** |
| **64** | **3.2729** | 3.2972 | **-0.0243** | **3.3542** | 3.4440 | **-0.0898** |
| **128** | **3.0555** | 3.0786 | **-0.0231** | **3.1440** | 3.2352 | **-0.0912** |
| **256** | **2.9151** | 2.9391 | **-0.0241** | **3.0093** | 3.1024 | **-0.0931** |
| **512** | **2.8445** | 2.8702 | **-0.0257** | **2.9434** | 3.0376 | **-0.0942** |
| **1024** | **2.7937** | 2.8202 | **-0.0265** | **2.8939** | 2.9902 | **-0.0963** |
| **2048** | **2.7594** | 2.7877 | **-0.0283** | **2.8608** | 2.9587 | **-0.0979** |
| **4096** | **2.7481** | 2.7778 | **-0.0297** | **2.8501** | 2.9495 | **-0.0994** |

---

## 2. 350M Hyperparameter Sweeps (600M Tokens per Arm)

Evaluated across 10-arm hyperparameter grids exploring base LR, Muon LR, and cautious masking on 600M tokens each:

| Architecture | Optimizer | Winning Arm | Base LR | Muon LR | Held-Out NLL | Held-Out PPL | Delta vs. Standard | Relative PPL Reduction |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Standard LLaMA** | **Muon + AdamW** | `std_350m_muon_adamw_base_s42` | 0.0015 | 0.020 | 3.0625 | 21.38 | *baseline* | *baseline* |
| **Algebraic Transformer** | **Muon + AdamW** | `alg_350m_muon_adamw_canonical_s42` | 0.0020 | 0.020 | **3.0423** | **20.95** | **-0.0203** | **+2.01%** |
| **Standard LLaMA** | **Hybrid Muon + C-AdamW** | `std_350m_hybrid_muon_cadamw_base_s42` | 0.0015 | 0.020 | 3.0613 | 21.36 | *baseline* | *baseline* |
| **Algebraic Transformer** | **Hybrid Muon + C-AdamW** | `alg_350m_hybrid_muon_cadamw_canonical_s42` | 0.0020 | 0.020 | **3.0436** | **20.98** | **-0.0177** | **+1.75%** |

---

## 3. 350M Main Benchmark Results (7.0B Tokens / Seed 42)

Full-scale evaluation on FineWeb-Edu (7.0B tokens / 6,675 steps per run, batch size 512, sequence length 2,048 across 16 Cloud TPU v4 chips) using the optimal sweep hyperparameters:

| Architecture | Optimizer | Held-Out NLL (nats) | Held-Out Perplexity | Final Train Loss | Delta NLL vs. Standard | Relative PPL Reduction | Throughput | Runtime |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Standard LLaMA** | **Muon + AdamW** | 2.5036 | 12.23 | 2.5085 | *baseline* | *baseline* | 574k tok/s | 3.39 hrs |
| **Algebraic Transformer** | **Muon + AdamW** | **2.4909** | **12.07** | **2.4956** | **-0.0128** | **+1.27%** | 517k tok/s | 3.76 hrs |
| **Standard LLaMA** | **Hybrid Muon + C-AdamW** | 2.5066 | 12.26 | 2.5118 | *baseline* | *baseline* | 574k tok/s | 3.39 hrs |
| **Algebraic Transformer** | **Hybrid Muon + C-AdamW** | **2.4943** | **12.11** | **2.4994** | **-0.0123** | **+1.22%** | 517k tok/s | 3.76 hrs |

---

## 4. Key Takeaways

1. **Consistent Scaling Superiority:** The Algebraic Transformer outperforms the Standard Transformer across all tested optimizers at both 125M and 350M scales.
2. **Learning Rate Forgiveness:** The Algebraic Transformer safely tolerates aggressive learning rates (`lr = 0.0020 - 0.0030`) without gradient explosion, whereas the Standard Transformer requires more constrained rates (`lr ≤ 0.0015`).
3. **No Representation Capacity Loss:** Despite completely eliminating all transcendental functions ($\exp, \ln, \sin, \cos$), the Algebraic Transformer achieves strictly lower training loss and held-out perplexity.
4. **Muon + AdamW Synergy:** Standard decoupled AdamW on the embedding matrix complements Muon's 2D polar matrix updates better than C-AdamW at large batch sizes (1.05M tokens/step), avoiding long-tail coordinate starvation.

---

## 5. Benchmark Artifacts & Reports

Full metric outputs, raw evaluation logs, and analysis documents are in [`results/`](results/):
* **350M Main Benchmark Report (7B tokens):** [`results/main_350m_report.md`](results/main_350m_report.md)
* **350M Benchmark Summary JSON:** [`results/main_350m_summary.json`](results/main_350m_summary.json)
* **350M Sweep Comparison Report:** [`results/hparam_sweep_350m_comparison_report.md`](results/hparam_sweep_350m_comparison_report.md)
* **350M Algebraic Sweep (10 arms):** [`results/hparam_sweep_350m_algebraic_report.md`](results/hparam_sweep_350m_algebraic_report.md)
* **350M Standard Sweep (10 arms):** [`results/hparam_sweep_350m_standard_report.md`](results/hparam_sweep_350m_standard_report.md)
* **125M Architecture Benchmark Report (5 seeds):** [`results/architecture_comparison_125m_report.md`](results/architecture_comparison_125m_report.md)
* **125M Algebraic Summary JSON:** [`results/main_125m_summary.json`](results/main_125m_summary.json)
* **125M Standard Summary JSON:** [`results/main_125m_standard_summary.json`](results/main_125m_standard_summary.json)

---

## 6. Reproduction

```bash
# Setup environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e . --no-deps

# Run unit tests on CPU
JAX_PLATFORMS=cpu pytest -v tests/

# Execute 16-chip TPU slice run
python3 scripts/run_slice.py runs/main_runs_350m/jobs.json \
  --group main_runs_350m \
  --hosts w0,w1,w2,w3 \
  --skip-done
```
