# 350M Architecture Head-to-Head Comparison Report: Algebraic vs. Standard

**Date Completed:** 2026-10-29 14:21:01 IST  
**Cluster Hardware:** Cloud TPU v4-32 (16 TPUs across 4 workers: `w0`, `w1`, `w2`, `w3`)  
**Scale:** 350M Active Parameters (`~332.5M` parameters)  
**Evaluation Token Budget:** 600M tokens per arm (573 steps, batch 512, micro 2, seq 2,048)  

## Head-to-Head Winning Results Table

| Architecture | Optimizer | Best Arm | Base LR | Muon LR | Held-Out NLL | Held-Out PPL | Delta NLL vs Standard | Relative PPL Reduction |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Standard LLaMA** | **muon_adamw** | `std_350m_muon_adamw_base_s42` | 0.0015 | 0.02 | 3.0625 | 21.38 | baseline | baseline |
| **Algebraic Transformer** | **muon_adamw** | `alg_350m_muon_adamw_canonical_s42` | 0.002 | 0.02 | **3.0423** | **20.95** | **-0.0203** | **+2.01%** |
| **Standard LLaMA** | **hybrid_muon_cadamw** | `std_350m_hybrid_muon_cadamw_base_s42` | 0.0015 | 0.02 | 3.0613 | 21.36 | baseline | baseline |
| **Algebraic Transformer** | **hybrid_muon_cadamw** | `alg_350m_hybrid_muon_cadamw_canonical_s42` | 0.002 | 0.02 | **3.0436** | **20.98** | **-0.0177** | **+1.75%** |
