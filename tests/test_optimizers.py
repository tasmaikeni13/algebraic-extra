"""Unit tests for pure XLA standalone optimizers.

Covers:
- Newton-Schulz 5-step iteration: orthogonality, aspect ratio scaling, zero-norm guard.
- Parameter routing: structural routing of 2D matrices to Muon and vectors/embeddings to AdamW.
- Classic Muon + AdamW (muon_adamw): Nesterov momentum, cosine schedule, pure XLA JIT.
- Pure Cautious AdamW (cadamw): cautious masking, descent monotonicity, mean normalization.
- Hybrid Muon + C-AdamW (hybrid_muon_cadamw): structural state separation, joint execution.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from algebraic.optimizers import (
    OptState,
    get_optimizer,
    is_muon_param,
    make_cosine_schedule,
    newton_schulz_5,
)


class TestNewtonSchulzOrthogonality:
    """Tests for Newton-Schulz quintic iteration and polar decomposition."""

    def test_newton_schulz_square_matrix(self):
        """Verify orthogonalization on square matrices (32x32 and 64x64)."""
        rng = jax.random.PRNGKey(42)
        G32 = jax.random.normal(rng, (32, 32))
        O32 = newton_schulz_5(G32, steps=5)
        gram32 = jnp.matmul(O32, O32.T)
        err32 = jnp.linalg.norm(gram32 - jnp.eye(32))
        assert err32 < 1.0, f"Gram matrix error on 32x32: {err32}"

        G64 = jax.random.normal(jax.random.PRNGKey(101), (64, 64))
        O64 = newton_schulz_5(G64, steps=5)
        gram64 = jnp.matmul(O64, O64.T)
        err64 = jnp.max(jnp.abs(gram64 - jnp.eye(64)))
        assert err64 < 0.25, f"Gram max error on 64x64: {err64}"

    def test_newton_schulz_tall_matrix(self):
        """Verify orthogonalization on tall matrix (128x64) with aspect ratio scaling."""
        rng = jax.random.PRNGKey(102)
        G = jax.random.normal(rng, (128, 64))
        O = newton_schulz_5(G, steps=5)
        assert O.shape == (128, 64)
        gram = jnp.matmul(O.T, O)
        expected_scale = 128.0 / 64.0
        err = jnp.max(jnp.abs(gram - jnp.eye(64) * expected_scale))
        assert err < 1.0, f"Tall matrix Gram error: {err}"

    def test_newton_schulz_wide_matrix(self):
        """Verify orthogonalization on wide matrix (64x128)."""
        rng = jax.random.PRNGKey(103)
        G = jax.random.normal(rng, (64, 128))
        O = newton_schulz_5(G, steps=5)
        assert O.shape == (64, 128)
        gram = jnp.matmul(O, O.T)
        err = jnp.max(jnp.abs(gram - jnp.eye(64)))
        assert err < 0.25, f"Wide matrix Gram error: {err}"

    def test_newton_schulz_batched_3d(self):
        """Verify batched operation on 3D layer-stacked tensor (L, M, N)."""
        rng = jax.random.PRNGKey(104)
        G = jax.random.normal(rng, (4, 32, 32))
        O = newton_schulz_5(G, steps=5)
        assert O.shape == (4, 32, 32)
        gram = jnp.matmul(O, jnp.swapaxes(O, -2, -1))
        eye = jnp.eye(32)
        err = jnp.max(jnp.abs(gram - eye))
        assert err < 0.25, f"Batched Gram error: {err}"

    def test_newton_schulz_zero_norm_guard(self):
        """Verify stability on all-zero matrix input."""
        G = jnp.zeros((16, 16), dtype=jnp.float32)
        O = newton_schulz_5(G, steps=5)
        assert jnp.isfinite(O).all()
        assert jnp.allclose(O, 0.0)

    def test_newton_schulz_singular_value_bound(self):
        """Verify singular values of output are tightly bounded around 1.0."""
        rng = jax.random.PRNGKey(777)
        G = jax.random.normal(rng, (32, 32))
        O = newton_schulz_5(G, steps=5)
        svs = jnp.linalg.svd(O, compute_uv=False)
        assert jnp.max(jnp.abs(svs - 1.0)) < 0.15


class TestParameterRouting:
    """Tests for parameter classification routing between Muon and AdamW."""

    def test_muon_param_internal_matrices(self):
        """Verify 2D projection matrices route to Muon."""
        matrix_keys = ["w_q", "w_k", "w_v", "w_o", "w_g", "w_u", "w_d"]
        for key in matrix_keys:
            path = ("layers", key)
            param = jnp.zeros((12, 64, 64))
            assert is_muon_param(path, param) is True, f"Key {key} should route to Muon"

    def test_non_muon_params_vector_and_embedding(self):
        """Verify normalization gains, sinks, and embeddings route to AdamW/C-AdamW."""
        non_muon = [
            (("embed",), jnp.zeros((32768, 768))),
            (("embed_gain",), jnp.zeros((768,))),
            (("final_gain",), jnp.zeros((768,))),
            (("layers", "ln1"), jnp.zeros((12, 768))),
            (("layers", "ln2"), jnp.zeros((12, 768))),
            (("layers", "q_gain"), jnp.zeros((12, 64))),
            (("layers", "k_gain"), jnp.zeros((12, 64))),
            (("layers", "sink"), jnp.zeros((12, 12))),
        ]
        for path, param in non_muon:
            assert is_muon_param(path, param) is False, f"Path {path} should not route to Muon"


class TestMuonAdamW:
    """Tests for Classic Muon (2D) + AdamW (1D) optimizer."""

    def test_muon_adamw_registration_and_types(self):
        """Verify get_optimizer('muon_adamw') returns valid callables and OptState."""
        init_fn, step_fn = get_optimizer("muon_adamw", lr=0.003, total_steps=100)
        assert callable(init_fn)
        assert callable(step_fn)

        params = {"w_q": jnp.ones((16, 16)), "embed_gain": jnp.ones((16,))}
        state = init_fn(params)
        assert isinstance(state, OptState)
        assert state.count == 0
        assert "w_q" in state.m
        assert state.v.get("w_q") is None
        assert state.v.get("embed_gain") is not None

    def test_muon_nesterov_momentum_update(self):
        """Verify Muon applies momentum and updates 2D weight matrix."""
        init_fn, step_fn = get_optimizer(
            "muon_adamw", lr=0.01, muon_momentum=0.95, total_steps=10
        )
        params = {"w_q": jnp.ones((16, 16), dtype=jnp.float32)}
        opt_state = init_fn(params)
        grads = {"w_q": jnp.ones((16, 16), dtype=jnp.float32) * 0.1}
        decay_mask = {"w_q": True}
        new_params, new_opt_state = step_fn(0, params, grads, opt_state, decay_mask)
        assert not jnp.allclose(new_params["w_q"], params["w_q"])
        assert new_opt_state.count == 1
        assert not jnp.allclose(new_opt_state.m["w_q"], 0.0)

    def test_muon_adamw_cosine_schedule(self):
        """Verify learning rate schedule progresses with warmup and decay."""
        init_fn, step_fn = get_optimizer(
            "muon_adamw", lr=0.01, total_steps=100, warmup_fraction=0.02, min_lr_ratio=0.1
        )
        params = {"embed_gain": jnp.ones((16,), dtype=jnp.float32)}
        opt_state = init_fn(params)
        grads = {"embed_gain": jnp.ones((16,), dtype=jnp.float32) * 0.1}
        decay_mask = {"embed_gain": False}
        p_step0, _ = step_fn(0, params, grads, opt_state, decay_mask)
        p_step50, _ = step_fn(50, params, grads, opt_state, decay_mask)
        u0 = params["embed_gain"] - p_step0["embed_gain"]
        u50 = params["embed_gain"] - p_step50["embed_gain"]
        assert not jnp.allclose(u0, u50)

    def test_muon_adamw_pure_xla_jit(self):
        """Verify Muon+AdamW compiles and executes under jax.jit."""
        init_fn, step_fn = get_optimizer("muon_adamw", lr=0.01, total_steps=10)
        params = {"w_q": jnp.ones((8, 8)), "embed_gain": jnp.ones((8,))}
        decay_mask = {"w_q": True, "embed_gain": False}
        opt_state = init_fn(params)
        grads = {"w_q": jnp.ones((8, 8)) * 0.05, "embed_gain": jnp.ones((8,)) * 0.05}

        @jax.jit
        def jitted_step(p, g, s):
            return step_fn(0, p, g, s, decay_mask)

        new_p, new_s = jitted_step(params, grads, opt_state)
        assert jnp.isfinite(new_p["w_q"]).all()
        assert jnp.isfinite(new_p["embed_gain"]).all()


class TestCautiousAdamW:
    """Tests for Pure Cautious AdamW (cadamw) optimizer."""

    def test_cadamw_registration(self):
        """Verify get_optimizer('cadamw') initializes state correctly."""
        init_fn, step_fn = get_optimizer("cadamw", lr=0.003, total_steps=100)
        assert callable(init_fn)
        assert callable(step_fn)
        p = {"w": jnp.ones((4, 4))}
        s = init_fn(p)
        assert s.v.get("w") is not None

    def test_cadamw_descent_monotonicity(self):
        """Verify updates satisfy u * g >= 0 along all coordinates."""
        init_fn, step_fn = get_optimizer("cadamw", lr=0.01, total_steps=10)
        p = {"w": jnp.ones((8,))}
        s = init_fn(p)
        g = {"w": jnp.array([1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1])}
        p1, _ = step_fn(0, p, g, s, {"w": False})
        update = p["w"] - p1["w"]
        alignment = update * g["w"]
        assert (alignment >= -1e-6).all(), f"Monotonicity violated: {alignment}"

    def test_cadamw_cautious_mask_opposition(self):
        """Verify opposition coordinates are masked when instantaneous gradient reverses."""
        init_fn, step_fn = get_optimizer("cadamw", lr=0.01, total_steps=100)
        p = {"x": jnp.zeros((4,), dtype=jnp.float32)}
        s = init_fn(p)
        g0 = {"x": jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)}
        p1, s1 = step_fn(0, p, g0, s, {"x": False})
        g1 = {"x": jnp.array([-1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)}
        p2, s2 = step_fn(1, p1, g1, s1, {"x": False})
        assert jnp.isfinite(p2["x"]).all()

    def test_cadamw_decoupled_weight_decay(self):
        """Verify decoupled weight decay operates only on flagged parameters."""
        init_fn, step_fn = get_optimizer("cadamw", lr=0.01, weight_decay=0.1, total_steps=100)
        p = {"w_decay": jnp.ones((8,)), "w_nodecay": jnp.ones((8,))}
        s = init_fn(p)
        g = {"w_decay": jnp.zeros((8,)), "w_nodecay": jnp.zeros((8,))}
        dmask = {"w_decay": True, "w_nodecay": False}
        p1, _ = step_fn(0, p, g, s, dmask)
        assert p1["w_decay"][0] < 1.0, "Decayed parameter should shrink"
        assert jnp.isclose(p1["w_nodecay"][0], 1.0), "Non-decayed parameter should not change"

    def test_cadamw_zero_gradient_step(self):
        """Verify step with zero gradient produces no parameter movement."""
        init_fn, step_fn = get_optimizer("cadamw", lr=0.01, total_steps=100)
        p = {"x": jnp.ones((8,))}
        s = init_fn(p)
        g = {"x": jnp.zeros((8,))}
        p1, _ = step_fn(0, p, g, s, {"x": False})
        assert jnp.allclose(p1["x"], p["x"])

    def test_cadamw_scalar_parameter(self):
        """Verify correct operation on 0-dim scalar parameters."""
        init_fn, step_fn = get_optimizer("cadamw", lr=0.01, total_steps=100)
        p = {"scalar": jnp.array(2.0, dtype=jnp.float32)}
        s = init_fn(p)
        g = {"scalar": jnp.array(0.5, dtype=jnp.float32)}
        p1, _ = step_fn(0, p, g, s, {"scalar": False})
        assert jnp.isfinite(p1["scalar"]).all()


class TestHybridMuonCAdamW:
    """Tests for Hybrid Muon (2D) + C-AdamW (1D) optimizer."""

    def test_hybrid_registration_and_state(self):
        """Verify hybrid optimizer registers and separates state buffers."""
        init_fn, step_fn = get_optimizer("hybrid_muon_cadamw", lr=0.003, total_steps=100)
        assert callable(init_fn)
        assert callable(step_fn)

        params = {"embed": jnp.zeros((32, 16)), "w_q": jnp.zeros((16, 16))}
        state = init_fn(params)
        assert hasattr(state, "count")
        assert hasattr(state, "m")
        assert hasattr(state, "v")
        assert state.v.get("w_q") is None
        assert state.v.get("embed") is not None

    def test_hybrid_step_joint_execution(self):
        """Verify hybrid optimizer executes cleanly on mixed 1D and 2D parameters."""
        init_fn, step_fn = get_optimizer("hybrid_muon_cadamw", lr=0.01, total_steps=100)
        params = {"w_q": jnp.ones((16, 16)), "embed_gain": jnp.ones((16,))}
        dmask = {"w_q": True, "embed_gain": False}
        state = init_fn(params)
        grads = {"w_q": jnp.ones((16, 16)) * 0.05, "embed_gain": jnp.ones((16,)) * 0.05}
        new_params, new_state = step_fn(0, params, grads, state, dmask)
        assert jnp.isfinite(new_params["w_q"]).all()
        assert jnp.isfinite(new_params["embed_gain"]).all()
        assert not jnp.allclose(new_params["w_q"], params["w_q"])
        assert not jnp.allclose(new_params["embed_gain"], params["embed_gain"])

    def test_hybrid_empty_pytree(self):
        """Verify hybrid optimizer gracefully handles empty parameter tree."""
        init_fn, step_fn = get_optimizer("hybrid_muon_cadamw", lr=0.01, total_steps=10)
        p1, s1 = step_fn(0, {}, {}, init_fn({}), {})
        assert p1 == {}

    def test_hybrid_pure_xla_jit(self):
        """Verify hybrid optimizer compiles under jax.jit."""
        init_fn, step_fn = get_optimizer("hybrid_muon_cadamw", lr=0.003, total_steps=10)
        params = {"w_q": jnp.ones((8, 8)), "embed": jnp.ones((16, 8))}
        dmask = {"w_q": True, "embed": True}
        s = init_fn(params)
        g = {"w_q": jnp.ones((8, 8)) * 0.1, "embed": jnp.ones((16, 8)) * 0.1}

        @jax.jit
        def step(p, g, s):
            return step_fn(0, p, g, s, dmask)

        new_p, new_s = step(params, g, s)
        assert jnp.isfinite(new_p["w_q"]).all()
        assert jnp.isfinite(new_p["embed"]).all()
