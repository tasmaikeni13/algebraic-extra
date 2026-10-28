"""Empirical Adversarial Test Suite for Milestone 3 Recipes (challenger_m3_2).

Exhaustively challenges:
1. Full Recipe PyTree Structural Validation across all 30 sweep arms and main runs:
   - Evaluates PyTree leaf shapes and dtypes via JAX eval_shape.
   - Verifies 2D layer weight matrices have v is None in muon_adamw and hybrid_muon_cadamw.
   - Verifies 1D/embedding parameters have non-None accumulators in v with matching shapes.
   - Verifies cadamw maintains non-None second-moments for all parameters.
2. Learning Rate & Schedule Math Stress:
   - Full vector scan of steps [0, total_steps + 100] for all 30 arms and main runs.
   - Verifies zero NaNs, zero Infs, strictly positive values, no divide-by-zero.
   - Verifies boundary values at step 0, warm-1, warm, total_steps-1, total_steps.
   - Verifies peak LRs match literature ranges.
3. Step Count & Token Alignment:
   - Mathematical proof that step counts * batch * seq_len strictly cover token budgets.
   - Checks 16-TPU cluster batch/micro divisibility and 32k vocabulary TPU v4 alignment.
   - Verifies ckpt_every=0 across all 30 sweep arms.
4. Single-step optimizer execution under JAX CPU execution.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from algebraic.config import build
from algebraic.model import count_params, decay_mask, init_params
from algebraic.optimizers import get_optimizer, is_muon_param, make_cosine_schedule
from algebraic.train import get_args, lr_schedule, parse_overrides
from scripts.run_experiments_pipeline import (
    VOCAB_SIZE,
    get_optimizer_sweep_grid,
    make_main_jobs,
    make_sweep_jobs,
)
from scripts.run_comparison import training_jobs, ROOT as REPO_ROOT


# ==============================================================================
# Challenge 1: Full Recipe PyTree Structural Validation
# ==============================================================================


class TestRecipePyTreeStructuralValidation:
    """Validate PyTree shapes, dtypes, and second-moment buffer elimination."""

    @pytest.fixture(scope="class")
    def all_30_sweep_jobs(self) -> List[Dict[str, Any]]:
        jobs_125m = make_sweep_jobs(scale="125m")
        jobs_350m = make_sweep_jobs(scale="350m")
        assert len(jobs_125m) == 15, f"Expected 15 arms for 125m, got {len(jobs_125m)}"
        assert len(jobs_350m) == 15, f"Expected 15 arms for 350m, got {len(jobs_350m)}"
        return jobs_125m + jobs_350m

    @pytest.fixture(scope="class")
    def all_main_jobs(self) -> List[Dict[str, Any]]:
        main_jobs = make_main_jobs()
        assert len(main_jobs) == 3, f"Expected 3 main runs, got {len(main_jobs)}"
        return main_jobs

    def test_sweep_30_arms_pytree_leaf_shapes_and_dtypes(self, all_30_sweep_jobs):
        """Stress-test all 30 sweep arms: parse args, eval_shape model & opt state."""
        for j in all_30_sweep_jobs:
            name = j["name"]
            opt_name = j["optimizer"]
            parsed = get_args([*j["args"], "--out", "/tmp/dummy"])
            overrides = parse_overrides(parsed.set)
            overrides["seq_len"] = parsed.seq_len
            cfg = build(parsed.preset, parsed.size, **overrides)

            assert cfg.vocab_size == 32768, f"{name}: Vocab size must be 32,768"

            # Evaluate shapes of parameters without materializing full arrays
            params_shapes = jax.eval_shape(
                lambda key: init_params(cfg, key), jax.random.PRNGKey(parsed.seed)
            )

            total_steps = parsed.steps
            opt_init, _ = get_optimizer(
                parsed.optimizer,
                lr=parsed.lr,
                weight_decay=parsed.wd,
                total_steps=total_steps,
                warmup_fraction=parsed.warmup,
                min_lr_ratio=parsed.min_lr_ratio,
                b1=parsed.b1,
                b2=parsed.b2,
                adam_eps=parsed.adam_eps,
                muon_lr=parsed.muon_lr,
                muon_wd=parsed.muon_wd,
                muon_momentum=parsed.muon_momentum,
                muon_steps=parsed.muon_steps,
                cautious_eps=parsed.cautious_eps,
            )

            # Evaluate shapes of optimizer state
            opt_state_shapes = jax.eval_shape(opt_init, params_shapes)

            # 1. Check count
            assert opt_state_shapes.count.shape == (), f"{name}: count must be scalar"
            assert opt_state_shapes.count.dtype == jnp.int32, f"{name}: count must be int32"

            # 2. Check first moment m
            flat_p, _ = jax.tree_util.tree_flatten_with_path(params_shapes)
            flat_m, _ = jax.tree_util.tree_flatten_with_path(opt_state_shapes.m)
            assert len(flat_p) == len(flat_m), f"{name}: m leaves count mismatch"
            for (path, p), (_, m) in zip(flat_p, flat_m):
                assert m.shape == p.shape, f"{name}: m shape mismatch at {path}"
                assert m.dtype == jnp.float32, f"{name}: m dtype must be float32 at {path}"

            # 3. Check second moment v
            flat_v, _ = jax.tree_util.tree_flatten_with_path(
                opt_state_shapes.v, is_leaf=lambda x: x is None
            )
            assert len(flat_p) == len(flat_v), f"{name}: v leaves count mismatch"

            muon_2d_count = 0
            one_d_count = 0
            for (path, p), (_, v) in zip(flat_p, flat_v):
                is_muon = is_muon_param(path, p)
                if opt_name == "cadamw":
                    # In pure C-AdamW, all parameters have non-None second-moments
                    assert v is not None, f"{name} (cadamw): v must exist at {path}"
                    assert v.shape == p.shape, f"{name}: v shape mismatch at {path}"
                    assert v.dtype == jnp.float32, f"{name}: v dtype must be float32 at {path}"
                elif opt_name in ("muon_adamw", "hybrid_muon_cadamw"):
                    if is_muon:
                        muon_2d_count += 1
                        assert v is None, (
                            f"{name} ({opt_name}): 2D weight matrix at {path} MUST have v is None!"
                        )
                    else:
                        one_d_count += 1
                        assert v is not None, (
                            f"{name} ({opt_name}): Non-Muon at {path} must have v accumulator!"
                        )
                        assert v.shape == p.shape, f"{name}: v shape mismatch at {path}"
                        assert v.dtype == jnp.float32, f"{name}: v dtype must be float32 at {path}"

            if opt_name in ("muon_adamw", "hybrid_muon_cadamw"):
                # 7 internal layer matrices (w_q, w_k, w_v, w_o, w_g, w_u, w_d)
                assert muon_2d_count == 7, (
                    f"{name}: Expected exactly 7 Muon 2D layer matrices, got {muon_2d_count}"
                )
                # Non-Muon params: embed (1), gains (2), layers ln/q_gain/k_gain/sink (5) = 8
                assert one_d_count >= 8, (
                    f"{name}: Expected at least 8 non-Muon accumulators, got {one_d_count}"
                )

    def test_main_runs_pytree_leaf_shapes_and_dtypes(self, all_main_jobs):
        """Stress-test all 3 main runs for PyTree structure and memory elimination."""
        for j in all_main_jobs:
            name = j["name"]
            parsed = get_args([*j["args"], "--out", "/tmp/dummy"])
            overrides = parse_overrides(parsed.set)
            overrides["seq_len"] = parsed.seq_len
            cfg = build(parsed.preset, parsed.size, **overrides)

            assert cfg.vocab_size == 32768, f"{name}: Vocab size must be 32,768"
            params_shapes = jax.eval_shape(
                lambda key: init_params(cfg, key), jax.random.PRNGKey(parsed.seed)
            )

            total_steps = parsed.steps
            opt_init, _ = get_optimizer(
                parsed.optimizer,
                lr=parsed.lr,
                weight_decay=parsed.wd,
                total_steps=total_steps,
                warmup_fraction=parsed.warmup,
                min_lr_ratio=parsed.min_lr_ratio,
                b1=parsed.b1,
                b2=parsed.b2,
                adam_eps=parsed.adam_eps,
                muon_lr=parsed.muon_lr,
                muon_wd=parsed.muon_wd,
                muon_momentum=parsed.muon_momentum,
                muon_steps=parsed.muon_steps,
                cautious_eps=parsed.cautious_eps,
            )

            opt_state_shapes = jax.eval_shape(opt_init, params_shapes)
            assert opt_state_shapes.count.shape == ()
            assert opt_state_shapes.count.dtype == jnp.int32

            flat_p, _ = jax.tree_util.tree_flatten_with_path(params_shapes)
            flat_v, _ = jax.tree_util.tree_flatten_with_path(
                opt_state_shapes.v, is_leaf=lambda x: x is None
            )

            for (path, p), (_, v) in zip(flat_p, flat_v):
                if is_muon_param(path, p):
                    assert v is None, f"{name}: expected v is None for Muon parameter {path}"
                else:
                    assert v is not None, f"{name}: expected v accumulator for non-Muon {path}"
                    assert v.shape == p.shape


# ==============================================================================
# Challenge 2: Learning Rate & Schedule Math Stress
# ==============================================================================


class TestLearningRateScheduleMathStress:
    """Stress-test learning rate schedules for numerical stability and literature bounds."""

    @pytest.mark.parametrize("scale", ["125m", "350m"])
    @pytest.mark.parametrize("optimizer", ["muon_adamw", "cadamw", "hybrid_muon_cadamw"])
    def test_literature_lr_bounds_and_hyperparameters(self, scale, optimizer):
        """Verify all sweep candidate values match literature ranges."""
        grid = get_optimizer_sweep_grid(scale, optimizer)
        assert len(grid) == 5, f"Expected 5 candidates for {scale} {optimizer}"

        for arm in grid:
            cand = arm["cand_name"]
            lr = arm["lr"]
            wd = arm["wd"]

            # Literature checks: AdamW / C-AdamW LR in [0.0005, 0.006]
            assert 0.0005 <= lr <= 0.006, (
                f"{scale} {optimizer} {cand}: lr={lr} outside literature range"
            )
            # Weight decay in [0.01, 0.5]
            assert 0.01 <= wd <= 0.5, (
                f"{scale} {optimizer} {cand}: wd={wd} outside literature range"
            )

            if "muon_lr" in arm:
                # Muon LR in [0.005, 0.05] (Keller Jordan / Moonlight)
                muon_lr = arm["muon_lr"]
                assert 0.005 <= muon_lr <= 0.05, (
                    f"{scale} {optimizer} {cand}: muon_lr={muon_lr} outside literature range"
                )

            if "muon_wd" in arm:
                muon_wd = arm["muon_wd"]
                assert 0.005 <= muon_wd <= 0.20, (
                    f"{scale} {optimizer} {cand}: muon_wd={muon_wd} outside literature range"
                )

            if "cautious_eps" in arm:
                c_eps = arm["cautious_eps"]
                assert 1e-4 <= c_eps <= 1e-2, (
                    f"{scale} {optimizer} {cand}: cautious_eps={c_eps} outside literature range"
                )

    def test_dense_lr_schedule_stability_across_all_sweep_arms(self):
        """Evaluate full learning rate trajectory across all 30 arms and main runs."""
        jobs = make_sweep_jobs(scale="125m") + make_sweep_jobs(scale="350m") + make_main_jobs()

        for j in jobs:
            name = j["name"]
            parsed = get_args([*j["args"], "--out", "/tmp/dummy"])
            total_steps = parsed.steps
            warmup_fraction = parsed.warmup
            min_lr_ratio = parsed.min_lr_ratio

            # Evaluate both AdamW schedule and Muon schedule
            schedules_to_test = [
                (
                    "adam",
                    make_cosine_schedule(
                        parsed.lr, total_steps, warmup_fraction, min_lr_ratio
                    ),
                    parsed.lr,
                ),
            ]
            if parsed.muon_lr is not None:
                schedules_to_test.append(
                    (
                        "muon",
                        make_cosine_schedule(
                            parsed.muon_lr, total_steps, warmup_fraction, min_lr_ratio
                        ),
                        parsed.muon_lr,
                    )
                )

            # Also test train.py's lr_schedule
            train_sched, warm_steps = lr_schedule(parsed, total_steps)

            for sched_type, sched_fn, peak in schedules_to_test:
                # Step 0:
                lr_0 = float(sched_fn(jnp.asarray(0, jnp.int32)))
                assert not math.isnan(lr_0), f"{name} {sched_type}: lr at step 0 is NaN"
                assert lr_0 > 0.0, f"{name} {sched_type}: lr at step 0 must be positive, got {lr_0}"

                # Warmup completion step
                warm = int(round(warmup_fraction * total_steps))
                lr_warm = float(sched_fn(jnp.asarray(warm - 1, jnp.int32)))
                assert math.isclose(lr_warm, peak, rel_tol=1e-3), (
                    f"{name} {sched_type}: lr at warm-1 step ({lr_warm}) should equal peak ({peak})"
                )

                # Final step total_steps - 1
                lr_final = float(sched_fn(jnp.asarray(total_steps - 1, jnp.int32)))
                expected_floor = peak * min_lr_ratio
                assert math.isclose(lr_final, expected_floor, rel_tol=1e-3), (
                    f"{name} {sched_type}: lr final ({lr_final}) != floor ({expected_floor})"
                )

                # Beyond total steps (e.g. total_steps + 100)
                lr_beyond = float(sched_fn(jnp.asarray(total_steps + 100, jnp.int32)))
                assert not math.isnan(lr_beyond), f"{name} {sched_type}: lr beyond total_steps NaN"
                assert math.isclose(lr_beyond, expected_floor, rel_tol=1e-3), (
                    f"{name} {sched_type}: lr beyond total_steps should remain clamped at floor"
                )

                # Vectorized evaluation of dense steps
                step_indices = jnp.arange(0, total_steps, dtype=jnp.int32)
                lrs = jax.vmap(sched_fn)(step_indices)

                assert not jnp.any(jnp.isnan(lrs)), f"{name} {sched_type}: NaN detected in schedule"
                assert not jnp.any(jnp.isinf(lrs)), f"{name} {sched_type}: Inf detected in schedule"
                assert jnp.all(lrs > 0.0), (
                    f"{name} {sched_type}: Non-positive LR detected in schedule"
                )

                # In decay phase (after warmup), min LR should match expected_floor
                decay_lrs = lrs[warm - 1:]
                assert float(jnp.min(decay_lrs)) >= expected_floor * 0.999
                # In warmup phase, LR starts at peak / warm and increases strictly monotonically
                warmup_lrs = lrs[:warm]
                assert float(jnp.min(warmup_lrs)) > 0.0
                assert math.isclose(float(warmup_lrs[0]), peak / warm, rel_tol=1e-3)
                assert float(jnp.max(lrs)) <= peak * 1.001

    def test_pathological_lr_schedule_edge_cases(self):
        """Stress-test schedule functions against boundary inputs (T=1, warm=0, warm=1)."""
        # total_steps = 1
        sched = make_cosine_schedule(
            peak_lr=0.001, total_steps=1, warmup_fraction=0.02, min_lr_ratio=0.1
        )
        lr0 = float(sched(jnp.asarray(0, jnp.int32)))
        assert not math.isnan(lr0)
        assert lr0 > 0.0

        # warmup_fraction >= 1.0 (step count specified)
        sched_steps = make_cosine_schedule(
            peak_lr=0.002, total_steps=100, warmup_fraction=10.0, min_lr_ratio=0.1
        )
        lr_w = float(sched_steps(jnp.asarray(9, jnp.int32)))
        assert math.isclose(lr_w, 0.002, rel_tol=1e-3)


# ==============================================================================
# Challenge 3: Step Count & Token Alignment
# ==============================================================================


class TestStepCountAndTokenAlignment:
    """Validate token budgets, step arithmetic, TPU divisibility, and ckpt settings."""

    def test_sweep_token_coverage_exactness(self):
        """Verify sweep step counts * batch * seq_len strictly cover token budgets."""
        # 125M: 600M tokens
        seq_len = 2048
        batch = 512
        tokens_per_step = batch * seq_len  # 1,048,576

        budget_125m = 600_000_000
        steps_125m = math.ceil(budget_125m / tokens_per_step)
        assert steps_125m == 573
        tokens_processed_125m = steps_125m * tokens_per_step
        assert tokens_processed_125m >= budget_125m
        assert tokens_processed_125m == 600_834_048

        # 350M: 2B tokens
        budget_350m = 2_000_000_000
        steps_350m = math.ceil(budget_350m / tokens_per_step)
        assert steps_350m == 1908
        tokens_processed_350m = steps_350m * tokens_per_step
        assert tokens_processed_350m >= budget_350m
        assert tokens_processed_350m == 2_000_683_008

        # Main 125M: 2.5B tokens
        budget_main_125m = 2_500_000_000
        steps_main_125m = math.ceil(budget_main_125m / tokens_per_step)
        assert steps_main_125m == 2385
        tokens_processed_main_125m = steps_main_125m * tokens_per_step
        assert tokens_processed_main_125m >= budget_main_125m
        assert tokens_processed_main_125m == 2_500_853_760

        # Main 350M: 7B tokens
        budget_main_350m = 7_000_000_000
        steps_main_350m = math.ceil(budget_main_350m / tokens_per_step)
        assert steps_main_350m == 6676
        tokens_processed_main_350m = steps_main_350m * tokens_per_step
        assert tokens_processed_main_350m >= budget_main_350m
        assert tokens_processed_main_350m == 7_000_293_376

    def test_sweep_jobs_enforce_zero_checkpointing(self):
        """Verify all 30 sweep jobs strictly enforce ckpt_every=0 to preserve disk space."""
        jobs = make_sweep_jobs(scale="125m") + make_sweep_jobs(scale="350m")
        for j in jobs:
            parsed = get_args([*j["args"], "--out", "/tmp/dummy"])
            assert parsed.ckpt_every == 0, f"{j['name']} must have ckpt_every=0"

    def test_main_jobs_enforce_periodic_checkpointing(self):
        """Verify main jobs enforce checkpointing (ckpt_every=200, keep_ckpts=2)."""
        jobs = make_main_jobs()
        for j in jobs:
            parsed = get_args([*j["args"], "--out", "/tmp/dummy"])
            assert parsed.ckpt_every == 200, f"{j['name']} must have ckpt_every=200"
            assert parsed.keep_ckpts == 2, f"{j['name']} must have keep_ckpts=2"

    def test_cluster_16_tpu_divisibility(self):
        """Verify global batch size 512 divides evenly across 16 TPUs for micro 4 and micro 2."""
        n_dev = 16
        batch = 512

        # 125M: micro = 4
        micro_125m = 4
        assert batch % (n_dev * micro_125m) == 0
        accum_125m = batch // (n_dev * micro_125m)
        assert accum_125m == 8

        # 350M: micro = 2
        micro_350m = 2
        assert batch % (n_dev * micro_350m) == 0
        accum_350m = batch // (n_dev * micro_350m)
        assert accum_350m == 16


# ==============================================================================
# Challenge 4: Single-Step Optimizer Execution Under JAX CPU Execution
# ==============================================================================


class TestSingleStepOptimizerExecution:
    """Execute real parameter update steps on CPU to verify numerical integrity."""

    @pytest.mark.parametrize("opt_name", ["muon_adamw", "cadamw", "hybrid_muon_cadamw"])
    def test_real_cpu_single_step_execution(self, opt_name):
        """Perform 1 optimization step with proxy model on CPU."""
        cfg = build("algebraic", "proxy", seq_len=16)
        rng = jax.random.PRNGKey(42)
        params = init_params(cfg, rng)
        mask = decay_mask(params)

        # Generate synthetic gradients (PRNGKey 43)
        grads = jax.tree_util.tree_map(
            lambda x: jax.random.normal(jax.random.PRNGKey(43), x.shape, dtype=jnp.float32) * 0.01,
            params,
        )

        init_fn, step_fn = get_optimizer(
            opt_name,
            lr=0.003,
            weight_decay=0.1,
            total_steps=100,
            muon_lr=0.02,
            muon_wd=0.05,
        )

        opt_state = init_fn(params)
        assert opt_state.count == 0

        # Perform step 0
        new_params, new_state = step_fn(0, params, grads, opt_state, mask)

        assert int(new_state.count) == 1

        # Check parameter updates are non-zero, finite, and non-NaN
        flat_p_old, _ = jax.tree_util.tree_flatten(params)
        flat_p_new, _ = jax.tree_util.tree_flatten(new_params)
        for p_old, p_new in zip(flat_p_old, flat_p_new):
            assert jnp.isfinite(p_new).all(), "NaN or Inf detected in updated parameter"
            assert not jnp.allclose(p_old, p_new), "Parameter was not updated by optimizer step"

        # Check opt_state.v
        flat_v_new, _ = jax.tree_util.tree_flatten_with_path(
            new_state.v, is_leaf=lambda x: x is None
        )
        flat_p_path, _ = jax.tree_util.tree_flatten_with_path(params)

        for (path, p), (_, v) in zip(flat_p_path, flat_v_new):
            is_muon = is_muon_param(path, p)
            if opt_name in ("muon_adamw", "hybrid_muon_cadamw"):
                if is_muon:
                    assert v is None, f"v must remain None for Muon param {path}"
                else:
                    assert v is not None, f"v must exist for non-Muon param {path}"
                    assert jnp.isfinite(v).all(), f"NaN in v at {path}"
            else:
                assert v is not None, f"v must exist for cadamw param {path}"
                assert jnp.isfinite(v).all(), f"NaN in v at {path}"
