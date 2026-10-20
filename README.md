# algebraic-extra

Hybrid and ARDS experiments for the 125M algebraic transformer.

This repository contains:
- **14 hybrid architectures** — combinatorial permutations of standard and algebraic transformer components
- **ARDS pipelines** — standard and algebraic transformers trained with AdamW + Algebraic Rational Decay Schedule
- **Pallas TPU kernels** — Flash Attention, QK RMS-norm, and fused loss heads for all 16 architectures

No pure-standard or pure-algebraic models live here. For the base implementations see [`algebraic-transformer`](https://github.com/tasmaikeni13/algebraic-transformer).

---

## Scale

All experiments target **125M parameters** on **2.5B tokens of FineWeb-Edu**, single seed (42).

| Setting | Value |
|---|---|
| Layers | 12 |
| d_model | 768 |
| Heads | 12 |
| d_ff | 2048 |
| Context | 2048 tokens |
| Vocab | 32,768 (BPE) |
| Batch | 512 seqs |
| Micro | 4 seqs/device |
| Steps | ~2,385 |
| Seed | 42 |

---

## Hybrid Architectures (`hybrid/`)

14 non-trivial permutations across 4 axes:

| Axis | Standard | Algebraic |
|---|---|---|
| Positional | RoPE (`rope`) | Geometric Cayley (`cayley_geo`) |
| FFN | SwiGLU (`swiglu`) | ALU-GLU β=0.5 (`alu_glu`) |
| Attention | Softmax (`softmax`) | Asinh-power + QK-norm + sink (`asinh`) |
| Loss Head | Cross-Entropy (`ce`) | Algebraic likelihood (`alg_ce`) |

Softmax-attention hybrids: **110,120,448 parameters**.
Asinh-attention hybrids: **110,122,128 parameters** (Δ = +1,680 for QK-norm gains and sink logits).

### All 14 hybrids

```
hybrid_rope_swiglu_softmax_alg_ce     hybrid_rope_swiglu_asinh_ce
hybrid_rope_swiglu_asinh_alg_ce       hybrid_rope_alu_softmax_ce
hybrid_rope_alu_softmax_alg_ce        hybrid_rope_alu_asinh_ce
hybrid_rope_alu_asinh_alg_ce          hybrid_cayley_swiglu_softmax_ce
hybrid_cayley_swiglu_softmax_alg_ce   hybrid_cayley_swiglu_asinh_ce
hybrid_cayley_swiglu_asinh_alg_ce     hybrid_cayley_alu_softmax_ce
hybrid_cayley_alu_softmax_alg_ce      hybrid_cayley_alu_asinh_ce
```

### Python API

```python
from hybrid.registry import list_presets, get_config

# List all 14 hybrids
names = list_presets()

# Build a config
cfg = get_config("hybrid_rope_alu_asinh_alg_ce", size="125m")
```

### CLI

```bash
python -m hybrid list
python -m hybrid info hybrid_cayley_swiglu_asinh_alg_ce
```

---

## ARDS Pipelines (`ards/`)

AdamW + Algebraic Rational Decay Schedule for **standard** and **pure algebraic** transformers.

Schedule (no cosine, no transcendentals):
```
lr(t) = peak / sqrt(1 + (t - t_warm)^2 / (0.8 T)^2)    for t > t_warm
lr(t) = peak * t / t_warm                                 for t <= t_warm
```

Defaults for 125M / 2.5B tokens: `peak=0.003`, `T=2385`, `t_warm=48`, `min_lr≈0.0019`.

### Python API

```python
from ards import standard_ards_config, algebraic_ards_config
from ards.schedule import ards_lr_schedule

cfg = standard_ards_config()       # Standard transformer + ARDS
cfg = algebraic_ards_config()      # Pure algebraic transformer + ARDS
lr_fn = ards_lr_schedule(peak=3e-3, total_steps=2385, warmup_steps=48)
```

### Training (single-chip smoke)

```bash
bash scripts/tpu_python.sh 0 -m algebraic.train \
  --preset standard --size 125m --schedule ards \
  --steps 2385 --batch 512 --micro 4 \
  --train ~/data/tok/fineweb_train_2.5B.npy \
  --valid ~/data/tok/fineweb_valid.npy \
  --seed 42 --out runs/standard_ards
```

---

## Pallas TPU Kernels (`src/algebraic/kernels/`)

| Kernel | File | Coverage |
|---|---|---|
| Flash Attention | `flash.py` | Softmax and Asinh-power, forward + backward, causal masking, sink logit |
| QK RMS-norm | `qk_norm.py` | Per-head RMS normalization fused before rotary |
| Fused Loss Head | `fused_head.py` | Tiled CE and algebraic likelihood, low HBM usage |
| Head-resident backward | `head_resident.py` | K/V-resident backward for large batch |

All kernels run in bfloat16 on Cloud TPU v4. Tiling: `flash_block=512`, `head_block_n=128`, `head_block_v=512`, `D=64`.

---

## Smoke Tests

Verify all 16 architectures (14 hybrids + 2 ARDS) with 1-step synthetic AdamW on TPU:

```bash
# Unified 16-architecture smoke runner
bash scripts/tpu_python.sh 0 scripts/smoke_all.py --all --steps 1 \
  --batch 4 --out /tmp/smoke_all.json

# With benchmark evaluation smoke (zero-shot + recall)
bash scripts/tpu_python.sh 0 scripts/smoke_all.py --all --eval-smoke \
  --out /tmp/smoke_all.json

# Single preset
bash scripts/tpu_python.sh 0 scripts/smoke_all.py \
  --preset hybrid_rope_alu_asinh_alg_ce --steps 1 --batch 4
```

Run the test suites:

```bash
# Hybrid parameter counts, forward/backward passes
bash scripts/tpu_python.sh 0 -m pytest tests/test_hybrids.py -v

# ARDS schedule precision, cosine rejection, single-step AdamW
bash scripts/tpu_python.sh 0 -m pytest tests/test_ards.py -v

# Unified 16-architecture smoke suite
bash scripts/tpu_python.sh 0 -m pytest tests/test_smoke_all.py -v

# Pallas TPU kernel regressions
bash scripts/tpu_python.sh 0 -m pytest tests/test_fused_head_tpu.py tests/test_qk_norm_tpu.py -v

# Algebraic component unit tests
bash scripts/tpu_python.sh 0 -m pytest tests/test_components.py -v
```

---

## Requirements

```
jax[tpu]>=0.6.0
flax
numpy
```

Install:
```bash
pip install -r requirements.txt
```

On TPU, use the helper:
```bash
bash scripts/tpu_python.sh 0 <your_script.py>
```

---

## Repository Layout

```
hybrid/          14 hybrid model configs and registry
ards/            ARDS training and evaluation pipelines
src/algebraic/
  algebraic.py   Algebraic primitives (rho, alu, cayley, asinh-attention, alg-likelihood)
  standard.py    Standard primitives (silu, rope, softmax-attention, cross-entropy)
  model.py       Unified configurable transformer (LMConfig)
  config.py      125m and proxy presets
  train.py       AdamW training loop with ARDS/cosine/WSD schedules
  kernels/       Pallas TPU kernels (flash, fused_head, qk_norm, head_resident)
configs/         comparison_125m.json
scripts/         smoke_all.py, smoke_comparison.py, smoke_benchmarks.py, tpu_python.sh
tests/           test_hybrids.py, test_ards.py, test_smoke_all.py, test_components.py,
                 test_training.py, test_flash_kernel_tpu.py, test_fused_head_tpu.py,
                 test_qk_norm_tpu.py
```
