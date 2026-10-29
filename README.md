# Algebraic Transformer (`algebraic-extra`)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hardware: Cloud TPU v4-32](https://img.shields.io/badge/Hardware-Cloud%20TPU%20v4--32-orange.svg)](https://cloud.google.com/tpu)
[![Framework: JAX / Pallas](https://img.shields.io/badge/Framework-JAX%20%7C%20Pallas-blue.svg)](https://github.com/google/jax)

A standalone, research-grade implementation and empirical evaluation of the **Algebraic Transformer** against the modern **Standard LLaMA-style Transformer** at **125M** and **350M** active parameters, trained from scratch on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) across Cloud TPU v4-32 pods (16 TPU chips across 4 host workers).

---

## The Core Question: Can Transformers Eliminate All Transcendentals?

In standard language models, non-linearities rely heavily on transcendental functions:
* **Attention:** Softmax computes exponentiation ($\exp(s_i)$) and normalizes by an exponential sum ($\sum_j \exp(s_j)$).
* **Positional Embeddings:** Rotary position embeddings (RoPE) evaluate trigonometric functions ($\sin$ and $\cos$).
* **Feed-Forward Blocks:** SwiGLU / GeGLU evaluate transcendentals via $\text{SiLU}(x) = x \cdot \sigma(x) = x / (1 + e^{-x})$.
* **Output Loss Head:** Cross-entropy evaluates log-sum-exp reductions requiring logarithms ($\ln$) and exponentials.

The **Algebraic Transformer** replaces every transcendental operation with purely algebraic primitives—functions composed strictly of field arithmetic ($+$, $-$, $\times$, $/$) and reciprocal square roots (`rsqrt`). Crucially, its **exact log-loss gradient evaluates in closed form without a single logarithm, exponential, or trigonometric function**.

---

## Architectural Comparison

| Component | Standard LLaMA-style Transformer | Pure Algebraic Transformer |
|---|---|---|
| **Attention Activation** | Softmax ($\exp(QK^T / \sqrt{d})$) | Algebraic asinh power contraction ($\rho(S / c)^n$) |
| **Attention Sink** | None | Learned per-head attention sink parameter |
| **Positional Encoding** | Trigonometric RoPE ($\sin$ / $\cos$ frequencies) | Rational Cayley geometric rotations |
| **Feed-Forward Network** | SwiGLU ($x \cdot \text{SiLU}(W_1 x)$) | ALU-GLU ($x \cdot \text{ALU}(W_1 x)$, rational algebraic root) |
| **Normalization** | Learned RMS gains | Learned RMS gains + QK normalization |
| **Output Distribution** | Softmax | Algebraic normalized power distribution ($\rho(z / 2)^8$) |
| **Objective Gradient** | Softmax Cross-Entropy | Exact algebraic log-loss gradient |

*Notation:* $\rho(x) = x + \sqrt{1 + x^2}$. The token embedding and output projection matrices are tied.  
The algebraic claim strictly covers the **model objective and its gradient** during training (the formal primitive audit is detailed in [`scripts/audit_algebraic.py`](scripts/audit_algebraic.py)).

---

## Why Eliminate Transcendentals? (Hardware Architecture Perspective)

In modern GPU and TPU silicon, transcendental functions cannot execute on dense matrix-multiply units (Tensor Cores / TPU Matrix Multiply Units [MXUs]). Instead, they are routed to **Special Function Units (SFUs)** that compute multi-cycle minimax/Chebyshev polynomial approximations or memory-mapped lookup tables:
1. **SFU Bottlenecks:** SFUs have **8× to 16× lower instruction throughput** per clock cycle than dense tensor cores.
2. **FlashAttention Thread Stalls:** In fused attention kernels, execution threads stall waiting on online softmax reduction barriers ($\exp(x - x_{\max})$).
3. **Pure Algebraic ASICs:** By proving that models train stably and achieve superior perplexity without transcendentals, this architecture paves the way for specialized algebraic accelerators:
   * **Die Area Reclamation:** Eliminating SFU silicon frees physical die area that can be dedicated to additional FP8/BF16 tensor cores or on-chip SRAM cache.
   * **Autonomous On-Chip Training:** Forward passes, backward passes, loss heads, and optimizer updates (AdamW uses only arithmetic and `rsqrt`) run entirely algebraically, removing host CPU synchronization bottlenecks.

---

## 125M Empirical Results (5 Random Seeds per Optimizer, 30 Runs Total)

Controlled comparison on FineWeb-Edu (2.5B tokens / 2,385 steps per run, batch size 512, sequence length 2,048 across 16 Cloud TPU v4 chips). Each optimizer was evaluated across **5 independent random seeds** (`42, 43, 44, 45, 46`):

### 5-Seed Head-to-Head Comparison Table

| Architecture | Optimizer | Held-Out NLL (nats) | Held-Out Perplexity | Final Train Loss | Delta NLL vs. Standard | Relative PPL Reduction |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Standard LLaMA** | **Muon + AdamW** | 2.8130 ± 0.0015 | 16.66 ± 0.02 | 2.7688 | *baseline* | *baseline* |
| **Algebraic Transformer** | **Muon + AdamW** | **2.7881 ± 0.0011** | **16.25 ± 0.02** | **2.7415** | **-0.0249** | **+2.46%** |
| **Standard LLaMA** | **Cautious AdamW (C-AdamW)** | 2.9847 ± 0.0042 | 19.78 ± 0.08 | 2.9425 | *baseline* | *baseline* |
| **Algebraic Transformer** | **Cautious AdamW (C-AdamW)** | **2.8868 ± 0.0035** | **17.94 ± 0.06** | **2.8423** | **-0.0979** | **+9.32%** |
| **Standard LLaMA** | **Hybrid Muon + C-AdamW** | 2.8149 ± 0.0018 | 16.69 ± 0.03 | 2.7704 | *baseline* | *baseline* |
| **Algebraic Transformer** | **Hybrid Muon + C-AdamW** | **2.7864 ± 0.0012** | **16.22 ± 0.02** | **2.7412** | **-0.0286** | **+2.81%** |

#### Overall 15-Run Grand Means:
* **Algebraic Transformer Grand Mean:** Held-Out NLL **2.8204**, Perplexity **16.80**
* **Standard LLaMA Grand Mean:** Held-Out NLL **2.8709**, Perplexity **17.71**
* **Grand Delta:** **-0.0504 nats** (**+5.13% relative perplexity reduction** for Algebraic Transformer across all runs).

### Context Scaling: Position-Wise NLL ($2^1$ to $2^{12}$ tokens)

Evaluated at each power-of-2 context position on held-out FineWeb-Edu validation data:

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

## 350M Empirical Results & Scaling Dynamics

At 350M scale ($D_{\text{model}}=1024, L=24, H=16, D_{\text{head}}=64, D_{\text{ff}}=2688$, active parameters: `332,453,248` non-embedding weights), models were evaluated across extensive 10-arm hyperparameter grids on 600M tokens each (20 arms total):

### 350M Head-to-Head Sweep Comparison

| Architecture | Optimizer | Best Winning Arm | Base LR | Muon LR | Held-Out NLL | Held-Out PPL | Delta NLL vs. Standard | Relative PPL Reduction |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Standard LLaMA** | **Muon + AdamW** | `std_350m_muon_adamw_base_s42` | 0.0015 | 0.020 | 3.0625 | 21.38 | *baseline* | *baseline* |
| **Algebraic Transformer** | **Muon + AdamW** | `alg_350m_muon_adamw_canonical_s42` | 0.0020 | 0.020 | **3.0423** | **20.95** | **-0.0203** | **+2.01%** |
| **Standard LLaMA** | **Hybrid Muon + C-AdamW** | `std_350m_hybrid_muon_cadamw_base_s42` | 0.0015 | 0.020 | 3.0613 | 21.36 | *baseline* | *baseline* |
| **Algebraic Transformer** | **Hybrid Muon + C-AdamW** | `alg_350m_hybrid_muon_cadamw_canonical_s42` | 0.0020 | 0.020 | **3.0436** | **20.98** | **-0.0177** | **+1.75%** |

### Key Findings at 350M Scale:
1. **Perplexity Superiority:** The Algebraic Transformer achieves lower held-out NLL and perplexity across both optimizers (**20.95 - 20.98** vs **21.36 - 21.38** for the Standard Transformer).
2. **Learning Rate Forgiveness:** The Algebraic Transformer safely tolerates aggressive learning rates (`lr=0.0020` to `0.0030`) without gradient explosion, converging to smoother loss basins. The Standard Transformer requires constrained learning rates (`lr=0.0015` or lower) and exhibits performance degradation at lower LRs (`lr=0.0005` yielded 23.10 - 25.54 PPL).
3. **Chinchilla-Optimal 7B Runs:** Main training runs on **7.0 Billion tokens** (6,675 steps) on FineWeb-Edu across all 16 TPUs are currently in progress.

---

## Standalone Pure XLA Optimizers

This repository includes standalone, pure JAX/XLA implementations of three state-of-the-art optimizers:
1. **Muon + AdamW (`muon_adamw`):**
   * Newton-Schulz matrix polar updates for 2D weight matrices (5 iteration steps).
   * Standard decoupled AdamW for 1D vectors, biases, and embedding tables.
2. **Cautious AdamW (`cadamw`):**
   * Incorporates directional sign-alignment masking ($\theta_{t+1} = \theta_t - \eta \cdot m_t / (\sqrt{v_t} + \epsilon) \cdot \mathbb{I}(g_t \cdot m_t > 0)$).
3. **Hybrid Muon + C-AdamW (`hybrid_muon_cadamw`):**
   * Routes 2D representations to Newton-Schulz polar updates while applying Cautious AdamW masking to embedding and normalization parameters.

---

## Benchmark Reports & Summaries

All raw metric outputs, logs, and markdown analysis documents are available under [`results/`](results/):
* **125M Architecture Benchmark Report (5 Seeds):** [`results/architecture_comparison_125m_report.md`](results/architecture_comparison_125m_report.md)
* **125M Algebraic Main Summary:** [`results/main_125m_summary.json`](results/main_125m_summary.json)
* **125M Standard Main Summary:** [`results/main_125m_standard_summary.json`](results/main_125m_standard_summary.json)
* **350M Architecture Head-to-Head Report:** [`results/hparam_sweep_350m_comparison_report.md`](results/hparam_sweep_350m_comparison_report.md)
* **350M Algebraic Sweep Report (10 Arms):** [`results/hparam_sweep_350m_algebraic_report.md`](results/hparam_sweep_350m_algebraic_report.md)
* **350M Standard Sweep Report (10 Arms):** [`results/hparam_sweep_350m_standard_report.md`](results/hparam_sweep_350m_standard_report.md)

---

## Environment & Reproduction

### Python Environment
Requires Python 3.10+ and JAX:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps

# Run unit tests on CPU
JAX_PLATFORMS=cpu pytest -v tests/
```

### Multi-Host Cloud TPU v4-32 Pod Slice
To execute distributed training across 16 TPU chips (4 worker hosts `w0`, `w1`, `w2`, `w3`):

```bash
# 1. Sync workers
bash scripts/setup_workers.sh w0 w1 w2 w3

# 2. Run data-parallel slice
python3 scripts/run_slice.py runs/hparam_sweep_350m_algebraic/jobs.json \
  --group hparam_sweep_350m_algebraic \
  --hosts w0,w1,w2,w3 \
  --skip-done
```

---

## Acknowledgments

Special thanks to Google's **TPU Research Cloud (TRC)** program for providing the Cloud TPU v4-32 compute resources used to train and evaluate these models.

---

## Citation

```bibtex
@software{keni2026algebraic,
  author = {Keni, Tasmai},
  title = {Algebraic Transformer: Causal Language Models with Algebraic Attention, Cayley Rotations, and Algebraic Cross-Entropy},
  year = {2026},
  url = {https://github.com/tasmaikeni13/algebraic-extra}
}
```
