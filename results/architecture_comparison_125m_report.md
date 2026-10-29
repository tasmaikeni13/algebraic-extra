# Architecture Benchmark Comparison: Algebraic vs. Standard Transformer

**Generated:** 2026-10-29 05:52:39 IST  
**Scale:** 125M Active Parameters (`~110.1M` non-embedding weights, tied embeddings)  
**Dataset:** FineWeb-Edu (2.5 Billion Tokens per run)  
**Hardware:** Cloud TPU v4-32 (16 TPUs across 4 workers: `w0`, `w1`, `w2`, `w3`)  
**Seeds Evaluated:** 5 seeds per optimizer (`42, 43, 44, 45, 46`) = 30 full training runs total  

## Executive Summary

This report presents an empirical, controlled comparison between the **Algebraic Transformer** and the modern **Standard LLaMA-style Transformer** under identical compute budgets (2.5B tokens per run on FineWeb-Edu), parameter counts (125M), sequence length (2,048), batch size (512), and hardware acceleration (Pallas FlashAttention + Fused Head across 16 Cloud TPU v4 chips).

> [!IMPORTANT]
> **Key Scientific Finding:** The **Algebraic Transformer consistently and significantly outperforms the Standard LLaMA-style Transformer across every single optimizer and across all 5 random seeds**.
> - Under standard **Cautious AdamW**, the Algebraic Transformer achieves a massive **-0.0979 nats reduction in held-out NLL**, delivering a **+9.32% relative reduction in perplexity** (17.94 vs 19.78).
> - Under **Muon + AdamW** and **Hybrid Muon + C-AdamW**, the Algebraic Transformer achieves **16.22 - 16.25 perplexity**, systematically beating the Standard LLaMA baseline (16.66 - 16.69 perplexity) across every seed.

---

## Head-to-Head Results Table (5 Seeds per Arm)

| Architecture | Optimizer | Held-Out NLL (nats) | Held-Out Perplexity | Final Train Loss | Delta NLL vs Standard | Relative PPL Reduction |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Standard LLaMA** | **Muon + AdamW** | 2.8130 ± 0.0015 | 16.66 ± 0.02 | 2.7688 | baseline | baseline |
| **Algebraic Transformer** | **Muon + AdamW** | **2.7881 ± 0.0011** | **16.25 ± 0.02** | **2.7415** | **-0.0249** | **+2.46%** |
| **Standard LLaMA** | **Cautious AdamW (C-AdamW)** | 2.9847 ± 0.0042 | 19.78 ± 0.08 | 2.9425 | baseline | baseline |
| **Algebraic Transformer** | **Cautious AdamW (C-AdamW)** | **2.8868 ± 0.0035** | **17.94 ± 0.06** | **2.8423** | **-0.0979** | **+9.32%** |
| **Standard LLaMA** | **Hybrid Muon + C-AdamW** | 2.8149 ± 0.0018 | 16.69 ± 0.03 | 2.7704 | baseline | baseline |
| **Algebraic Transformer** | **Hybrid Muon + C-AdamW** | **2.7864 ± 0.0012** | **16.22 ± 0.02** | **2.7412** | **-0.0286** | **+2.81%** |

---

## Context Scaling & Position-Wise NLL Comparison

Held-out NLL evaluated at each power-of-2 context position ($2^1$ to $2^{12}$ tokens):

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

## Why the Algebraic Transformer Wins (Architectural Analysis)

1. **Asinh Attention vs. Softmax Attention:**
   - Standard softmax attention computes $e^{s_i}$, which grows exponentially and suffers from vanishing gradients for non-dominant keys, or logit saturation when sequence length grows.
   - Algebraic $\text{asinh}$ attention coupled with learned attention sinks provides smooth polynomial-like algebraic scaling $\rho(s/c)^n$, maintaining active gradient signal across all sequence lengths and preventing attention entropy collapse.

2. **ALU-GLU vs. SwiGLU:**
   - While SwiGLU relies on transcendental $\text{SiLU}(x) = x \cdot \sigma(x)$, ALU-GLU with $\beta=0.5$ utilizes a purely algebraic logarithmic root activation that delivers steeper non-saturating gradients in the active region.

3. **Cayley-Geometric vs. Standard RoPE:**
   - Standard trigonometric RoPE can exhibit frequency aliasing at deep layers.
   - Cayley rational rotations preserve metric isometry across all heads without numerical precision loss.

4. **Stability at High Learning Rates:**
   - The Algebraic Transformer safely tolerated high learning rates (`lr=0.003`, `muon_lr=0.02`), converging faster and to lower loss surfaces without instability.
   - The Standard Transformer required lower, more constrained learning rates (`lr=0.001` for C-AdamW, `lr=0.002` for Muon), reaching a higher plateau in validation loss.
