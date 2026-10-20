# Release Verification & Acceptance Report (TEST_READY)

**Repository**: `algebraic-transformer`  
**Status**: VERIFIED & PUBLICATION READY  
**Architecture Specialization**: 125M Causal Language Model ($V=32,768, L=12, d=768, d_{\text{ff}}=2048, H=12, D=64$)  
**Integrity Mode**: Development / Local Verification  
**Evaluation Hardware**: Google Cloud TPU v4 (Single Chip Execution)  

---

## 1. Acceptance Test Suite Execution Command

To run the authoritative end-to-end acceptance suite across all 4 verification tiers:
```bash
bash scripts/tpu_python.sh 0 -m pytest tests/test_e2e_acceptance.py -v
```

To run individual verification tiers:
```bash
# Tier 1: Repository Cleanliness, Pruning, Absence of 350M/1B, Safe Stubs
bash scripts/tpu_python.sh 0 -m pytest tests/test_e2e_acceptance.py -m tier1 -v

# Tier 2: 125M Architecture Contracts, SIZES, Tokenizer, Vocabulary
bash scripts/tpu_python.sh 0 -m pytest tests/test_e2e_acceptance.py -m tier2 -v

# Tier 3: Hybrid Permutations, ARDS Purity, Pallas TPU Kernels, Parameter Counts
bash scripts/tpu_python.sh 0 -m pytest tests/test_e2e_acceptance.py -m tier3 -v

# Tier 4: Unified Smoke Execution, Zero-Shot & Recall Benchmarks, Safety Audit
bash scripts/tpu_python.sh 0 -m pytest tests/test_e2e_acceptance.py -m tier4 -v
```

To run dedicated component and smoke suites:
```bash
# Unified 16-Architecture Smoke Test Suite
bash scripts/tpu_python.sh 0 -m pytest tests/test_smoke_all.py -v

# All 14 Hybrid Permutations & Parameter Counts
bash scripts/tpu_python.sh 0 -m pytest tests/test_hybrids.py -v

# ARDS Schedule Precision, Cosine Rejection & Single-Step AdamW
bash scripts/tpu_python.sh 0 -m pytest tests/test_ards.py -v

# Pallas TPU Kernel Regression Suites
bash scripts/tpu_python.sh 0 -m pytest tests/test_fused_head_tpu.py tests/test_qk_norm_tpu.py -v
```

---

## 2. CLI Smoke Runner Commands

```bash
# 1. Unified 16-Architecture Synthetic Smoke Test (1-step AdamW on TPU)
bash scripts/tpu_python.sh 0 scripts/smoke_all.py --all --steps 1 --batch 4 --out results/verification/smoke_all.json

# 2. Unified 16-Architecture Smoke Test with Benchmark Evaluation Smoke
bash scripts/tpu_python.sh 0 scripts/smoke_all.py --all --eval-smoke --out results/verification/smoke_all.json

# 3. Benchmark Evaluation Smoke Test (8 Public Tasks + 4 Recall Tasks on local frozen data)
bash scripts/tpu_python.sh 0 scripts/smoke_benchmarks.py \
  --benchmarks /home/tasma/data/evals \
  --recall-benchmarks /home/tasma/data/recall_llama32k \
  --out results/verification/benchmark_smoke.json

# 4. 125M Comparison Model Synthetic Smoke Test
bash scripts/tpu_python.sh 0 scripts/smoke_comparison.py --scale 125m --out results/verification/smoke_125m.json
```

---

## 3. Comprehensive Acceptance Verification Checklist

### R1. Repository Pruning & 125M Specialization
- [x] All 350M and 1B scale configs (`configs/comparison_350m.json`, `configs/comparison_1b.json`, `configs/sweep_*.json`) are deleted.
- [x] Scale-specific scripts (`run_experiments_pipeline_350m.py`, `prepare_fineweb_edu_1b.py`, `hparam_sweep.py`) are deleted.
- [x] `SIZES` in `src/algebraic/config.py` contains strictly `proxy` and `125m`.
- [x] `configs/` contains strictly `comparison_125m.json` configured for 2.5B tokens, batch 512, micro 4, seq_len 2048, seed 42.
- [x] Active codebase contains zero active definitions of 350M/1B scales.

### R2. Hybrid Component Permutations (`hybrid/`)
- [x] Dedicated `hybrid/` module implements all 14 non-trivial hybrid configurations spanning the 4 axes.
- [x] Pure standard and pure algebraic architectures are strictly excluded from `hybrid/`.
- [x] All 7 Softmax hybrids have exactly 110,120,448 parameters.
- [x] All 7 Asinh hybrids have exactly 110,122,128 parameters ($\Delta = +1,680$).
- [x] Hybrid presets are accessible via CLI, Python API, and `algebraic.config.build()`.

### R3. ARDS Purity Pipelines (`ards/`)
- [x] Dedicated `ards/` module contains training and evaluation pipelines for Standard+ARDS and Algebraic+ARDS.
- [x] ARDS rational decay schedule implements $\text{lr}(t) = \text{peak} / \sqrt{1 + (t - t_{\text{warm}})^2 / (0.8 T)^2}$.
- [x] Cosine learning rate schedules are strictly prohibited and rejected with `ValueError`.
- [x] AST audit confirms zero transcendental functions (`cos`, `sin`, `scipy`) in `ards/`.
- [x] Pipelines default to single-seed 42 initialization and data batching.

### R4. Pallas TPU Kernel Integration
- [x] Pallas TPU kernels (`flash_attention`, `fused_head`, `qk_rms_norm`) operate reliably across all 14 hybrids and ARDS variants.
- [x] Forward and backward passes execute with finite positive loss and finite non-zero bfloat16 gradients without NaNs.
- [x] Hardware tiling parameters ($flash\_block=512$, $head\_block\_n=128$, $head\_block\_v=512$, $D=64$) align with TPU v4 MXU lanes.
- [x] Traced JAXPR gradient graph contains zero forbidden transcendentals on algebraic sub-paths.

### R5. Comprehensive Smoke Testing & Safety
- [x] Fast 1-step synthetic AdamW smoke test verifies all 16 architectures on TPU (`scripts/smoke_all.py`).
- [x] Zero-shot evaluation smoke executes across all 8 public tasks without missing data or runtime failures.
- [x] Recall benchmark smoke executes across all 4 bounded-context tasks with zero evidence truncation.
- [x] No full 2.5B-token training runs are launched; steps $> 100$ rejected in development mode.
- [x] Remote upload scripts (`upload_to_hf.py`, `push_to_hf.py`) safely stubbed to exit 0 without network activity.
- [x] Remote Git pushes are disabled.

---

## 4. Verification Artifacts Matrix

| Artifact Path | Producing Script | Verified Key Outputs | Status |
|---------------|------------------|----------------------|--------|
| `results/verification/smoke_all.json` | `scripts/smoke_all.py` | 16 models passed, 1 step, finite loss & bfloat16 gradients, exact param counts | VALIDATED |
| `results/verification/benchmark_smoke.json` | `scripts/smoke_benchmarks.py` | 8 public tasks, 4 recall tasks, 0 evidence truncation | VALIDATED |
| `results/verification/smoke_125m.json` | `scripts/smoke_comparison.py` | Standard (110,120,448) & Algebraic (110,122,128), primitive audit passed | VALIDATED |
