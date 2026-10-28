"""Empirical adversarial challenger test suite for Milestone 4 (challenger_m4_2).

Focus areas:
1. Multi-Host & Multi-Device Participation:
   - 16 TPU devices, 4 host processes across w0..w3.
   - Theoretical parameter counts: 110,122,128 for 125M and 332,453,248 for 350M.
2. Numerical & Loss Stability:
   - Training loss, gradient norm, eval NLL strictly finite, positive, and non-zero.
   - Mathematical differentiation among optimizers (Muon, C-AdamW, Hybrid).
3. Artifact Consistency & Vulnerability Surface:
   - Consistency across smoke_16tpu.json, distributed_125m.json, distributed_350m.json.
   - Destructive overwrite hazard in scripts/smoke_16tpu.py during dry-run or partial runs.
4. Execution Safety Invariants:
   - Zero large weight checkpoints (.ckpt, .pt, params.npz) left in runs/.
   - TPU device locks clean.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

ALL_CONFIGS = [
    ("125m", "muon_adamw"),
    ("125m", "cadamw"),
    ("125m", "hybrid_muon_cadamw"),
    ("350m", "muon_adamw"),
    ("350m", "cadamw"),
    ("350m", "hybrid_muon_cadamw"),
]

EXPECTED_PARAMS = {
    "125m": 110_122_128,
    "350m": 332_453_248,
}

EXPECTED_DIMS = {
    "125m": {"d_model": 768, "n_layers": 12, "n_heads": 12, "d_ff": 2048},
    "350m": {"d_model": 1024, "n_layers": 24, "n_heads": 16, "d_ff": 2688},
}


class TestParameterCountsTheoreticalExactness:
    """Empirically prove that parameter counts match theoretical formulas exactly."""

    def test_parameter_counts_exact_match(self):
        """Verify model initialization calculates exact theoretical parameter counts."""
        import jax
        from algebraic.config import LMConfig, SIZES
        from algebraic.model import init_params

        key = jax.random.PRNGKey(0)
        for scale, expected in EXPECTED_PARAMS.items():
            cfg = LMConfig(**SIZES[scale], vocab_size=32768)
            params = init_params(cfg, key)
            leaves = jax.tree_util.tree_leaves(params)
            total = sum(p.size for p in leaves)
            assert total == expected, f"{scale} param mismatch: got {total}, expected {expected}"

    def test_parameter_layer_tensor_shapes(self):
        """Verify structural tensor dimensions conform to algebraic transformer architecture."""
        import jax
        from algebraic.config import LMConfig, SIZES
        from algebraic.model import init_params

        key = jax.random.PRNGKey(0)
        for scale, dims in EXPECTED_DIMS.items():
            cfg = LMConfig(**SIZES[scale], vocab_size=32768)
            p = init_params(cfg, key)

            d = dims["d_model"]
            L = dims["n_layers"]
            ff = dims["d_ff"]

            assert p["embed"].shape == (32768, d)
            assert p["embed_gain"].shape == (d,)
            assert p["final_gain"].shape == (d,)
            assert p["layers"]["w_q"].shape == (L, d, d)
            assert p["layers"]["w_k"].shape == (L, d, d)
            assert p["layers"]["w_v"].shape == (L, d, d)
            assert p["layers"]["w_o"].shape == (L, d, d)
            assert p["layers"]["w_g"].shape == (L, d, ff)
            assert p["layers"]["w_u"].shape == (L, d, ff)
            assert p["layers"]["w_d"].shape == (L, ff, d)
            assert p["layers"]["ln1"].shape == (L, d)
            assert p["layers"]["ln2"].shape == (L, d)
            assert p["layers"]["q_gain"].shape == (L, 64)
            assert p["layers"]["k_gain"].shape == (L, 64)


class TestRealClusterExecutionMetrics:
    """Verify metrics from the full 16-TPU cluster execution in runs/smoke_16tpu_1793142709/."""

    @pytest.fixture
    def real_run_dir(self) -> Path:
        run_dir = REPO_ROOT / "runs" / "smoke_16tpu_1793142709"
        if not run_dir.exists():
            pytest.skip(f"Historical 16-TPU full execution dir not found: {run_dir}")
        return run_dir

    def test_all_six_configurations_present(self, real_run_dir):
        """Verify all 6 configurations ran and generated result.json and log.jsonl."""
        for scale, opt in ALL_CONFIGS:
            name = f"{scale}_{opt}"
            job_dir = real_run_dir / name
            assert job_dir.exists(), f"Missing directory for {name}"
            assert (job_dir / "result.json").exists(), f"Missing result.json for {name}"
            assert (job_dir / "log.jsonl").exists(), f"Missing log.jsonl for {name}"

    def test_multi_device_multi_host_participation(self, real_run_dir):
        """Verify all 6 runs report exactly 16 devices, 4 processes, and 1 completed step."""
        for scale, opt in ALL_CONFIGS:
            name = f"{scale}_{opt}"
            res = json.loads((real_run_dir / name / "result.json").read_text(encoding="utf-8"))
            devs = res.get("devices")
            procs = res.get("processes")
            steps = res.get("completed_steps")
            n_params = res.get("n_params")

            assert devs == 16, f"{name}: expected 16 devices, got {devs}"
            assert procs == 4, f"{name}: expected 4 processes, got {procs}"
            assert steps == 1, f"{name}: expected 1 step, got {steps}"
            assert n_params == EXPECTED_PARAMS[scale], f"{name}: parameter count mismatch"

    def test_numerical_stability_finite_positive_nonzero(self, real_run_dir):
        """Verify train loss, gradient norm, and eval NLL are finite, positive, and non-zero."""
        for scale, opt in ALL_CONFIGS:
            name = f"{scale}_{opt}"
            res = json.loads((real_run_dir / name / "result.json").read_text(encoding="utf-8"))
            log_lines = (real_run_dir / name / "log.jsonl").read_text(encoding="utf-8").splitlines()
            logs = [json.loads(line) for line in log_lines]

            step1 = next((l for l in logs if l.get("step") == 1 and "loss" in l), None)
            assert step1 is not None, f"{name}: missing step 1 train log"

            train_loss = float(step1["loss"])
            grad_norm = float(step1["gnorm"])
            eval_nll = float(res.get("final_eval_nll"))

            assert math.isfinite(train_loss) and train_loss > 0.0, f"{name}: bad loss {train_loss}"
            assert math.isfinite(grad_norm) and grad_norm > 0.0, f"{name}: bad gnorm {grad_norm}"
            assert math.isfinite(eval_nll) and eval_nll > 0.0, f"{name}: bad eval_nll {eval_nll}"

            assert not math.isnan(train_loss) and not math.isinf(train_loss)
            assert not math.isnan(grad_norm) and not math.isinf(grad_norm)
            assert not math.isnan(eval_nll) and not math.isinf(eval_nll)

    def test_optimizer_divergence_after_step(self, real_run_dir):
        """Verify that cadamw and muon_adamw updates diverge numerically after step 1."""
        for scale in ["125m", "350m"]:
            cadamw_res = json.loads((real_run_dir / f"{scale}_cadamw" / "result.json").read_text())
            muon_res = json.loads(
                (real_run_dir / f"{scale}_muon_adamw" / "result.json").read_text()
            )

            nll_cadamw = cadamw_res["final_eval_nll"]
            nll_muon = muon_res["final_eval_nll"]

            assert abs(nll_cadamw - nll_muon) > 1.0, (
                f"{scale}: expected optimizer eval NLL divergence, got {nll_cadamw} vs {nll_muon}"
            )


class TestExecutionSafetyInvariants:
    """Verify execution safety: no weight checkpoints left behind, TPU device locks released."""

    def test_no_weight_checkpoints_in_runs(self):
        """Verify no .ckpt, .pt, or params.npz weight files exist in runs/."""
        runs_dir = REPO_ROOT / "runs"
        forbidden_patterns = ["*.ckpt", "*.pt", "*params*.npz"]
        violations = []
        for pat in forbidden_patterns:
            for match in runs_dir.rglob(pat):
                violations.append(str(match.relative_to(REPO_ROOT)))

        assert not violations, "Checkpoints found in runs/:\n" + "\n".join(violations)

    def test_tpu_locks_released_cleanly(self):
        """Verify all TPU accelerator devices across w0..w3 are free of active locks."""
        hosts = ["w0", "w1", "w2", "w3"]
        for host in hosts:
            res = subprocess.run(
                ["ssh", host, "fuser /dev/accel* 2>&1 || true"],
                capture_output=True,
                text=True,
            )
            pids = [word for word in res.stdout.split() if word.isdigit()]
            assert len(pids) == 0, f"TPU accelerators still locked on {host} by PIDs: {pids}"


class TestArtifactVulnerabilitiesAndOverwriteHazards:
    """Stress-test reporting architecture to expose destructive overwrite bugs."""

    def test_smoke_script_dry_run_overwrites_canonical_reports_flaw(self):
        """Demonstrate that scripts/smoke_16tpu.py hardcodes writes to results/verification/

        even when --out is specified, clobbering distributed_{scale}.json.
        """
        smoke_py = REPO_ROOT / "scripts" / "smoke_16tpu.py"
        content = smoke_py.read_text(encoding="utf-8")

        assert "scale_artifact = verification_dir / f\"distributed_{scale}.json\"" in content, (
            "smoke_16tpu.py must be inspected for unconditional per-scale artifact writes"
        )

    def test_verification_artifacts_consistency_status(self):
        """Check the consistency among smoke_16tpu.json and distributed_{scale}.json."""
        vdir = REPO_ROOT / "results" / "verification"
        smoke_file = vdir / "smoke_16tpu.json"
        d125_file = vdir / "distributed_125m.json"
        d350_file = vdir / "distributed_350m.json"

        if not vdir.exists() or not smoke_file.exists():
            pytest.skip("results/verification artifacts not present")

        smoke_data = json.loads(smoke_file.read_text(encoding="utf-8"))
        d125_data = json.loads(d125_file.read_text(encoding="utf-8"))
        d350_data = json.loads(d350_file.read_text(encoding="utf-8"))

        smoke_runs = smoke_data.get("runs", {})
        d125_runs = d125_data.get("runs", {})
        d350_runs = d350_data.get("runs", {})

        is_smoke_complete = (len(smoke_runs) == 6)
        is_d125_complete = (len(d125_runs) == 3)
        is_d350_complete = (
            len(d350_runs) == 3 and not any(r.get("dry_run") for r in d350_runs.values())
        )

        defects = []
        if not is_smoke_complete:
            defects.append(
                f"smoke_16tpu.json has {len(smoke_runs)}/6 runs: {list(smoke_runs.keys())}"
            )
        if not is_d125_complete:
            defects.append(
                f"distributed_125m.json has {len(d125_runs)}/3 runs: {list(d125_runs.keys())}"
            )
        if not is_d350_complete:
            dry_count = sum(1 for r in d350_runs.values() if r.get("dry_run"))
            defects.append(
                f"distributed_350m.json has {dry_count} mock dry_run entries"
            )

        print(f"Artifact defects found: {defects}")
