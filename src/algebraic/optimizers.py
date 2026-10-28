"""Pure XLA standalone optimizers: Muon + AdamW, C-AdamW, and Hybrid Muon + C-AdamW.

Provides faithful, accelerator-native implementations of:
1. Classic Muon (2D weights with 5-step Newton-Schulz iteration, Frobenius pre-conditioning,
   aspect ratio scaling, and Nesterov momentum) + AdamW (1D weights, cosine schedule).
2. Pure Cautious AdamW (C-AdamW) with update alignment masking and mean normalization.
3. Hybrid Muon (2D weights) + C-AdamW (1D weights) with structural parameter routing
   and state memory reduction.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp


class OptState(NamedTuple):
    """Optimizer state tracking step count, first-moment m, and second-moment v."""

    count: jax.Array
    m: Any
    v: Any


def newton_schulz_5(G: jax.Array, steps: int = 5, eps: float = 1e-7) -> jax.Array:
    """Pure XLA quintic Newton-Schulz iteration with polar refinement.

    Operates on the trailing two dimensions (..., M, N). If M > N, transposes to (..., N, M)
    so Gram matrices are min(M, N) x min(M, N). Applies pre-normalization by Frobenius norm,
    5 quintic iteration steps with Keller Jordan coefficients, Schulz order-3 polar refinement,
    and aspect ratio scaling sqrt(max(1, M/N)).
    """
    if G.ndim < 2:
        return G

    a, b, c = 3.4445, -4.7750, 2.0315
    M, N = G.shape[-2], G.shape[-1]
    transposed = M > N
    X = jnp.swapaxes(G, -2, -1) if transposed else G

    # Pre-normalization by Frobenius norm along matrix dimensions
    norm = jnp.sqrt(jnp.sum(jnp.square(X), axis=(-2, -1), keepdims=True) + eps)
    X = X / norm

    # Quintic Newton-Schulz polynomial iteration
    for _ in range(steps):
        A = jnp.matmul(X, jnp.swapaxes(X, -2, -1))
        A2 = jnp.matmul(A, A)
        B = b * A + c * A2
        X = a * X + jnp.matmul(B, X)

    # Schulz order-3 polar refinement steps to ensure spectral convergence
    min_dim = min(M, N)
    eye = jnp.eye(min_dim, dtype=X.dtype)
    for _ in range(2):
        A = jnp.matmul(X, jnp.swapaxes(X, -2, -1))
        X = 0.5 * jnp.matmul(3.0 * eye - A, X)

    if transposed:
        X = jnp.swapaxes(X, -2, -1)

    scale = jnp.sqrt(jnp.maximum(1.0, float(M) / float(N)))
    return X * scale


def is_muon_param(path: Any, param: Any) -> bool:
    """Classify whether parameter should be updated by Muon or AdamW/C-AdamW.

    Muon handles internal 2D layer weight matrices (w_q, w_k, w_v, w_o, w_g, w_u, w_d).
    Embeddings, normalization gains, attention sinks, and biases route to AdamW/C-AdamW.
    """
    if path:
        last = path[-1] if isinstance(path, (list, tuple)) else path
        key = getattr(last, "key", str(last))
        key_str = str(key)
        if key_str == "embed":
            return False
        if key_str.startswith("w_"):
            return True
        if (
            key_str.startswith("ln")
            or key_str.endswith("_gain")
            or key_str == "sink"
            or "bias" in key_str
        ):
            return False
    return getattr(param, "ndim", 0) >= 2


def make_cosine_schedule(
    peak_lr: float,
    total_steps: int,
    warmup_fraction: float,
    min_lr_ratio: float,
) -> Callable[[jax.Array], jax.Array]:
    """Build pure XLA cosine learning rate schedule with linear warmup."""
    peak = peak_lr
    warm = (
        float(warmup_fraction)
        if warmup_fraction >= 1
        else max(1.0, float(round(warmup_fraction * total_steps)))
    )
    floor = peak * min_lr_ratio
    total_t = float(total_steps)

    def schedule(step: jax.Array) -> jax.Array:
        t = jnp.asarray(step, jnp.float32) + 1.0
        warm_f = jnp.minimum(1.0, t / warm)
        prog = jnp.clip((t - warm) / jnp.maximum(1.0, total_t - warm), 0.0, 1.0)
        lr = floor + (peak - floor) * 0.5 * (1.0 + jnp.cos(jnp.pi * prog))
        return jnp.where(t < warm, peak * warm_f, lr)

    return schedule


def get_optimizer(
    name: str,
    lr: float,
    weight_decay: float = 0.1,
    total_steps: int = 1000,
    warmup_fraction: float = 0.02,
    min_lr_ratio: float = 0.1,
    b1: float = 0.9,
    b2: float = 0.95,
    adam_eps: float = 1e-8,
    muon_lr: float | None = None,
    muon_wd: float | None = None,
    muon_momentum: float = 0.95,
    muon_steps: int = 5,
    cautious_eps: float = 1e-3,
    **kwargs: Any,
) -> Tuple[
    Callable[[Any], OptState],
    Callable[[int | jax.Array, Any, Any, OptState, Any], Tuple[Any, OptState]],
]:
    """Factory returning (init_fn, step_fn) for pure XLA optimizers.

    Supported optimizer names:
    - 'muon_adamw': Classic Muon on 2D weights + AdamW on 1D weights with cosine schedule.
    - 'cadamw': Pure Cautious AdamW on all parameters with update alignment masking.
    - 'hybrid_muon_cadamw': Muon on 2D weights + C-AdamW on 1D weights with memory savings.
    """
    opt_name = name.lower().strip()
    valid_names = ("muon_adamw", "cadamw", "hybrid_muon_cadamw")
    if opt_name not in valid_names:
        raise ValueError(f"Unknown optimizer: {name}. Choices: {valid_names}")

    actual_muon_lr = lr if muon_lr is None else muon_lr
    actual_muon_wd = weight_decay if muon_wd is None else muon_wd

    adam_schedule = make_cosine_schedule(lr, total_steps, warmup_fraction, min_lr_ratio)
    muon_schedule = make_cosine_schedule(actual_muon_lr, total_steps, warmup_fraction, min_lr_ratio)

    def init_fn(params: Any) -> OptState:
        count = jnp.zeros([], dtype=jnp.int32)
        m = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)
        if opt_name == "cadamw":
            v = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)
        else:
            def make_v(path: Any, p: Any) -> Any:
                if is_muon_param(path, p):
                    return None
                return jnp.zeros_like(p)

            v = jax.tree_util.tree_map_with_path(make_v, params)
        return OptState(count=count, m=m, v=v)

    def step_fn(
        step: int | jax.Array,
        params: Any,
        grads: Any,
        opt_state: OptState,
        decay_mask: Any,
    ) -> Tuple[Any, OptState]:
        step_arr = jnp.asarray(step, dtype=jnp.int32)
        adam_lr_t = adam_schedule(step_arr)
        muon_lr_t = muon_schedule(step_arr)
        t = jnp.asarray(step_arr + 1, dtype=jnp.float32)
        c1 = 1.0 - b1 ** t
        c2 = 1.0 - b2 ** t

        is_leaf_none = lambda x: x is None
        flat_p, treedef = jax.tree_util.tree_flatten_with_path(params, is_leaf=is_leaf_none)
        flat_g, _ = jax.tree_util.tree_flatten_with_path(grads, is_leaf=is_leaf_none)
        flat_m, _ = jax.tree_util.tree_flatten_with_path(opt_state.m, is_leaf=is_leaf_none)
        flat_v, _ = jax.tree_util.tree_flatten_with_path(opt_state.v, is_leaf=is_leaf_none)
        flat_d, _ = jax.tree_util.tree_flatten_with_path(decay_mask, is_leaf=is_leaf_none)

        new_p_list = []
        new_m_list = []
        new_v_list = []

        for (path, p), (_, g), (_, m), (_, v), (_, d) in zip(
            flat_p, flat_g, flat_m, flat_v, flat_d
        ):
            use_muon = (opt_name in ("muon_adamw", "hybrid_muon_cadamw")) and is_muon_param(path, p)
            if use_muon:
                m_new = muon_momentum * m + (1.0 - muon_momentum) * g
                v_nesterov = (1.0 - muon_momentum) * g + muon_momentum * m_new
                o = newton_schulz_5(v_nesterov, steps=muon_steps)
                decay_fac = jnp.where(d, 1.0 - muon_lr_t * actual_muon_wd, 1.0)
                p_new = p * decay_fac - muon_lr_t * o
                new_p_list.append(p_new)
                new_m_list.append(m_new)
                new_v_list.append(None)
            elif opt_name in ("cadamw", "hybrid_muon_cadamw"):
                m_new = b1 * m + (1.0 - b1) * g
                v_new = b2 * v + (1.0 - b2) * (g * g)
                u = (m_new / c1) / (jnp.sqrt(v_new / c2) + adam_eps)
                mask = jnp.where(u * g > 0.0, 1.0, 0.0)
                mask_mean = jnp.mean(mask)
                mask_norm = mask / jnp.maximum(mask_mean, cautious_eps)
                u_cautious = u * mask_norm
                decay_fac = jnp.where(d, 1.0 - adam_lr_t * weight_decay, 1.0)
                p_new = p * decay_fac - adam_lr_t * u_cautious
                new_p_list.append(p_new)
                new_m_list.append(m_new)
                new_v_list.append(v_new)
            else:
                m_new = b1 * m + (1.0 - b1) * g
                v_new = b2 * v + (1.0 - b2) * (g * g)
                u = (m_new / c1) / (jnp.sqrt(v_new / c2) + adam_eps)
                decay_fac = jnp.where(d, 1.0 - adam_lr_t * weight_decay, 1.0)
                p_new = p * decay_fac - adam_lr_t * u
                new_p_list.append(p_new)
                new_m_list.append(m_new)
                new_v_list.append(v_new)

        new_params = treedef.unflatten(new_p_list)
        new_state = OptState(
            count=opt_state.count + 1,
            m=treedef.unflatten(new_m_list),
            v=treedef.unflatten(new_v_list),
        )
        return new_params, new_state

    return init_fn, step_fn


__all__ = [
    "OptState",
    "newton_schulz_5",
    "is_muon_param",
    "make_cosine_schedule",
    "get_optimizer",
]
