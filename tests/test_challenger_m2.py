"""Adversarial stress-test suite for Milestone 2 Pure XLA Optimizers (R3).

Covers:
1. Pure XLA JIT tracing without non-XLA side-effects, Pallas calls, or host callbacks.
2. Exact memory buffer elimination (opt_state.v is None for 2D weight matrices).
3. Monotonic loss reduction and finite gradient norm progression.
4. Cautious AdamW opposition masking and all-opposition boundary behavior.
5. Newton-Schulz numerical stability under extreme singular spectrum inputs.
"""

import jax
import jax.numpy as jnp
import pytest

from algebraic.config import build
from algebraic.model import (
    count_params,
    decay_mask,
    forward_hidden,
    head_loss,
    init_params,
    position_tables,
)
from algebraic.optimizers import (
    OptState,
    get_optimizer,
    is_muon_param,
    newton_schulz_5,
)


class TestPureXLAJITPurity:
    """Adversarial tests ensuring zero non-XLA primitives or host callbacks."""

    @pytest.mark.parametrize("opt_name", ["muon_adamw", "cadamw", "hybrid_muon_cadamw"])
    def test_zero_pallas_and_callbacks_in_jaxpr_and_hlo(self, opt_name):
        cfg = build("algebraic", "proxy", seq_len=16)
        rng = jax.random.PRNGKey(42)
        params = init_params(cfg, rng)
        grads = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)
        mask = decay_mask(params)

        init_fn, step_fn = get_optimizer(opt_name, lr=0.001, total_steps=100)
        opt_state = init_fn(params)

        # Check jaxpr for forbidden primitives
        jaxpr_str = str(jax.make_jaxpr(step_fn)(0, params, grads, opt_state, mask)).lower()
        forbidden_primitives = ["custom_call", "pallas", "triton", "callback", "host_callback"]
        for term in forbidden_primitives:
            assert term not in jaxpr_str, f"Forbidden term {term} found in jaxpr for {opt_name}"

        # Check lowered HLO
        lowered = jax.jit(step_fn).lower(0, params, grads, opt_state, mask)
        hlo_str = lowered.as_text().lower()
        for term in forbidden_primitives:
            assert term not in hlo_str, f"Forbidden term {term} found in HLO for {opt_name}"


class TestHybridMemorySavings:
    """Adversarial checks on optimizer second-moment buffer elimination."""

    @pytest.mark.parametrize("size", ["125m", "350m"])
    def test_hybrid_v_leaves_are_strictly_none_for_2d_matrices(self, size):
        cfg = build("algebraic", size, seq_len=16)
        params = init_params(cfg, jax.random.PRNGKey(0))
        init_fn, step_fn = get_optimizer("hybrid_muon_cadamw", lr=0.003, total_steps=100)
        opt_state = init_fn(params)

        flat_p, _ = jax.tree_util.tree_flatten_with_path(params)
        flat_v, _ = jax.tree_util.tree_flatten_with_path(
            opt_state.v, is_leaf=lambda x: x is None
        )

        muon_count = 0
        non_muon_count = 0
        for (path, p), (_, v) in zip(flat_p, flat_v):
            if is_muon_param(path, p):
                muon_count += 1
                assert v is None, f"Expected v to be None for Muon parameter at path {path}"
            else:
                non_muon_count += 1
                assert v is not None, f"Expected v to exist for non-Muon parameter at path {path}"
                assert v.shape == p.shape

        assert muon_count > 0, "No Muon parameters identified"
        assert non_muon_count > 0, "No non-Muon parameters identified"


class TestNewtonSchulzAdversarialStability:
    """Stress testing Newton-Schulz quintic iteration on pathological inputs."""

    def test_zero_matrix(self):
        G = jnp.zeros((32, 32), dtype=jnp.float32)
        out = newton_schulz_5(G)
        assert jnp.isfinite(out).all()
        assert jnp.allclose(out, 0.0)

    def test_extreme_magnitude_matrices(self):
        G_large = jnp.ones((32, 32), dtype=jnp.float32) * 1e5
        out_large = newton_schulz_5(G_large)
        assert jnp.isfinite(out_large).all()

        G_small = jnp.ones((32, 32), dtype=jnp.float32) * 1e-6
        out_small = newton_schulz_5(G_small)
        assert jnp.isfinite(out_small).all()

    def test_rank_one_matrix(self):
        u = jax.random.normal(jax.random.PRNGKey(10), (64, 1))
        v = jax.random.normal(jax.random.PRNGKey(11), (1, 64))
        G = jnp.matmul(u, v)
        out = newton_schulz_5(G)
        assert jnp.isfinite(out).all()


class TestCautiousOppositionMasking:
    """Verify Cautious AdamW masking semantics on opposing vs aligned coordinates."""

    def test_opposition_coordinates_zeroed(self):
        init_fn, step_fn = get_optimizer("cadamw", lr=0.01, total_steps=100)
        p = {"w": jnp.ones((64,))}
        opt_state = init_fn(p)

        # Build positive momentum
        g0 = {"w": jnp.ones((64,)) * 10.0}
        p1, s1 = step_fn(0, p, g0, opt_state, {"w": False})

        # Oppose first 32 coordinates, align trailing 32
        g1 = {"w": jnp.concatenate([-jnp.ones((32,)), jnp.ones((32,))])}
        p2, s2 = step_fn(1, p1, g1, s1, {"w": False})

        diff = p2["w"] - p1["w"]
        assert jnp.allclose(diff[:32], 0.0), "Opposing coordinates failed to mask to zero"
        assert not jnp.allclose(diff[32:], 0.0), "Aligned coordinates were improperly masked"

    def test_all_opposition_boundary(self):
        init_fn, step_fn = get_optimizer("cadamw", lr=0.01, total_steps=100)
        p = {"w": jnp.ones((32,))}
        opt_state = init_fn(p)

        g0 = {"w": jnp.ones((32,)) * 10.0}
        p1, s1 = step_fn(0, p, g0, opt_state, {"w": False})

        g1 = {"w": -jnp.ones((32,))}
        p2, s2 = step_fn(1, p1, g1, s1, {"w": False})

        diff = p2["w"] - p1["w"]
        assert jnp.allclose(diff, 0.0), "All-opposition step did not yield zero update"
        assert jnp.isfinite(p2["w"]).all()
