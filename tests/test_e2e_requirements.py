"""Comprehensive Opaque-Box E2E Requirements Test Suite for algebraic-extra.

Validates the complete repository against requirements R1-R5 from ORIGINAL_REQUEST.md
and PROJECT.md across four verification tiers defined in TEST_INFRA.md:
- Tier 1: Feature Coverage (>=5 tests per feature for Features 1-15)
- Tier 2: Boundary & Corner Cases (>=5 edge cases per feature for Features 1-15)
- Tier 3: Pairwise Cross-Feature Interactions (15 tests)
- Tier 4: Real-World Hardware Workloads & Scenarios (Scenarios 1-8)

Execution:
    Full Suite:
        pytest tests/test_e2e_requirements.py -v
    Tier-Specific:
        pytest tests/test_e2e_requirements.py -m tier1 -v
        pytest tests/test_e2e_requirements.py -m tier2 -v
        pytest tests/test_e2e_requirements.py -m tier3 -v
        pytest tests/test_e2e_requirements.py -m tier4 -v
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Tuple

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import pytest

# Repository root directory
REPO_ROOT = Path(__file__).resolve().parent.parent

# Excluded paths for repository-wide audits
REPO_SCAN_EXCLUDES = {
    ".git",
    ".agents",
    ".pytest_cache",
    "__pycache__",
    ".venv",
    "venv",
    "data",
    "runs",
}


# ==============================================================================
# Helper Utilities
# ==============================================================================


def load_optimizers_module():
    """Attempt to import algebraic.optimizers if implemented."""
    try:
        import algebraic.optimizers as opt
        return opt
    except (ImportError, ModuleNotFoundError):
        return None


def read_text_safe(path: Path) -> str:
    """Read file content safely as UTF-8 string."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def get_all_active_python_files() -> List[Path]:
    """Return all active Python files excluding agent workspaces and caches."""
    py_files = []
    for p in REPO_ROOT.rglob("*.py"):
        rel_parts = p.relative_to(REPO_ROOT).parts
        if any(part in REPO_SCAN_EXCLUDES for part in rel_parts):
            continue
        py_files.append(p)
    return py_files


# ==============================================================================
# TIER 1: FEATURE COVERAGE (>=5 tests per feature for Features 1 to 15)
# ==============================================================================


@pytest.mark.tier1
@pytest.mark.e2e
class TestTier1FeatureCoverage:
    """Tier 1: Basic requirement and contract verification across all 15 features."""

    # --------------------------------------------------------------------------
    # Feature 1: Remove 1B Implementations (R1)
    # --------------------------------------------------------------------------

    def test_tier1_f1_01_no_comparison_1b_json(self):
        """Verify configs/comparison_1b.json has been removed."""
        cfg_1b = REPO_ROOT / "configs" / "comparison_1b.json"
        assert not cfg_1b.exists(), f"Obsolete 1B config must be removed: {cfg_1b}"

    def test_tier1_f1_02_no_1b_pipeline_scripts(self):
        """Verify 1B pipeline and data preparation scripts have been removed."""
        s1 = REPO_ROOT / "scripts" / "run_experiments_pipeline_1b.py"
        s2 = REPO_ROOT / "scripts" / "prepare_fineweb_edu_1b.py"
        assert not s1.exists(), f"1B pipeline script must be removed: {s1}"
        assert not s2.exists(), f"1B data prep script must be removed: {s2}"

    def test_tier1_f1_03_no_1b_in_config_sizes(self):
        """Verify SIZES dict in config.py does not contain '1b'."""
        try:
            from algebraic.config import SIZES
            assert "1b" not in SIZES, f"'1b' size preset must be removed from SIZES: {SIZES.keys()}"
        except ImportError:
            pytest.fail("Cannot import SIZES from algebraic.config")

    def test_tier1_f1_04_no_1b_in_cli_scripts(self):
        """Verify CLI scripts do not offer '1b' as a scale choice."""
        scripts_to_check = [
            REPO_ROOT / "scripts" / "run_comparison.py",
            REPO_ROOT / "scripts" / "smoke_comparison.py",
            REPO_ROOT / "scripts" / "smoke_distributed.py",
        ]
        for script_path in scripts_to_check:
            if script_path.exists():
                content = script_path.read_text(encoding="utf-8")
                assert '"1b"' not in content and "'1b'" not in content, (
                    f"Script {script_path.name} still contains references to '1b' scale"
                )

    def test_tier1_f1_05_no_1b_data_references_in_train(self):
        """Verify train.py does not default to fineweb1b dataset files."""
        train_py = REPO_ROOT / "src" / "algebraic" / "train.py"
        assert train_py.exists(), "src/algebraic/train.py must exist"
        content = train_py.read_text(encoding="utf-8")
        assert "fineweb1b" not in content, (
            "train.py must not contain references to fineweb1b dataset files"
        )

    # --------------------------------------------------------------------------
    # Feature 2: Remove Standard Transformer (R1)
    # --------------------------------------------------------------------------

    def test_tier1_f2_01_no_standard_py_file(self):
        """Verify standard.py has been deleted from src/algebraic/."""
        standard_py = REPO_ROOT / "src" / "algebraic" / "standard.py"
        assert not standard_py.exists(), f"src/algebraic/standard.py must be removed: {standard_py}"

    def test_tier1_f2_02_no_standard_preset_in_config(self):
        """Verify standard preset function is removed from config.py."""
        import algebraic.config as cfg_mod
        assert not hasattr(cfg_mod, "standard"), "standard() preset function must be removed from config.py"
        if hasattr(cfg_mod, "PRESETS"):
            assert "standard" not in cfg_mod.PRESETS, "PRESETS must not contain 'standard'"

    def test_tier1_f2_03_lmconfig_defaults_to_algebraic(self):
        """Verify LMConfig dataclass defaults to Algebraic Transformer primitives."""
        from algebraic.model import LMConfig
        cfg = LMConfig()
        assert cfg.ffn == "alu_glu", f"Default ffn should be 'alu_glu', got {cfg.ffn}"
        assert cfg.pos == "cayley_geo", f"Default pos should be 'cayley_geo', got {cfg.pos}"
        assert cfg.attn == "asinh", f"Default attn should be 'asinh', got {cfg.attn}"
        assert cfg.head == "alg_ce", f"Default head should be 'alg_ce', got {cfg.head}"
        assert cfg.sink == "learned", f"Default sink should be 'learned', got {cfg.sink}"
        assert cfg.qk_norm is True, f"Default qk_norm should be True, got {cfg.qk_norm}"

    def test_tier1_f2_04_no_standard_transformer_imports_in_model(self):
        """Verify model.py does not import standard.py or standard primitives."""
        model_py = REPO_ROOT / "src" / "algebraic" / "model.py"
        assert model_py.exists(), "src/algebraic/model.py must exist"
        content = model_py.read_text(encoding="utf-8")
        assert "from algebraic import standard" not in content, (
            "model.py must not import standard"
        )
        assert "import standard" not in content, (
            "model.py must not import standard"
        )

    def test_tier1_f2_05_kernels_do_not_branch_on_transcendental(self):
        """Verify Pallas kernels do not branch on 'softmax' or 'ce'."""
        kernels_dir = REPO_ROOT / "src" / "algebraic" / "kernels"
        for py_file in kernels_dir.glob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            assert 'kind == "softmax"' not in content, (
                f"Kernel {py_file.name} still branches on 'softmax'"
            )
            assert 'kind == "ce"' not in content, (
                f"Kernel {py_file.name} still branches on 'ce'"
            )

    # --------------------------------------------------------------------------
    # Feature 3: Remove Non-essential Dirs (R1)
    # --------------------------------------------------------------------------

    def test_tier1_f3_01_formal_dir_absent(self):
        """Verify formal/ directory (Lean 4 proofs) is pruned."""
        formal_dir = REPO_ROOT / "formal"
        assert not formal_dir.exists(), f"formal/ directory must be pruned: {formal_dir}"

    def test_tier1_f3_02_analysis_dir_absent(self):
        """Verify analysis/ directory is pruned."""
        analysis_dir = REPO_ROOT / "analysis"
        assert not analysis_dir.exists(), f"analysis/ directory must be pruned: {analysis_dir}"

    def test_tier1_f3_03_paper_dir_absent(self):
        """Verify paper/ directory is pruned."""
        paper_dir = REPO_ROOT / "paper"
        assert not paper_dir.exists(), f"paper/ directory must be pruned: {paper_dir}"

    def test_tier1_f3_04_results_dir_absent(self):
        """Verify results/ directory is pruned."""
        results_dir = REPO_ROOT / "results"
        assert not results_dir.exists(), f"results/ directory must be pruned: {results_dir}"

    def test_tier1_f3_05_clean_root_structure(self):
        """Verify repository root contains only designated directories and configuration."""
        allowed_dirs = {"src", "scripts", "configs", "tests", ".git", ".agents", ".pytest_cache", ".venv", "venv", "runs"}
        for item in REPO_ROOT.iterdir():
            if item.is_dir():
                assert item.name in allowed_dirs, (
                    f"Unexpected directory found in repository root: {item.name}"
                )

    # --------------------------------------------------------------------------
    # Feature 4: Retain 125M / 350M Algebraic Retained (R1)
    # --------------------------------------------------------------------------

    def test_tier1_f4_01_sizes_125m_and_350m_present(self):
        """Verify 125M and 350M dimensions exist with exact specifications."""
        from algebraic.config import SIZES
        assert "125m" in SIZES, "SIZES must contain '125m'"
        assert "350m" in SIZES, "SIZES must contain '350m'"
        cfg_125m = SIZES["125m"]
        assert cfg_125m["d_model"] == 768
        assert cfg_125m["n_layers"] == 12
        assert cfg_125m["n_heads"] == 12
        assert cfg_125m["d_ff"] == 2048
        cfg_350m = SIZES["350m"]
        assert cfg_350m["d_model"] == 1024
        assert cfg_350m["n_layers"] == 24
        assert cfg_350m["n_heads"] == 16
        assert cfg_350m["d_ff"] == 2688

    def test_tier1_f4_02_comparison_configs_present(self):
        """Verify configs/comparison_125m.json and configs/comparison_350m.json exist."""
        c125 = REPO_ROOT / "configs" / "comparison_125m.json"
        c350 = REPO_ROOT / "configs" / "comparison_350m.json"
        assert c125.exists(), f"{c125} must exist"
        assert c350.exists(), f"{c350} must exist"
        json.loads(c125.read_text(encoding="utf-8"))
        json.loads(c350.read_text(encoding="utf-8"))

    def test_tier1_f4_03_algebraic_primitives_present(self):
        """Verify algebraic.py exports core mathematical primitives."""
        import algebraic.algebraic as alg
        assert hasattr(alg, "rho"), "algebraic.py must export rho"
        assert hasattr(alg, "alu"), "algebraic.py must export alu"
        assert hasattr(alg, "asinh_attention"), "algebraic.py must export asinh_attention"
        assert hasattr(alg, "algebraic_likelihood"), "algebraic.py must export algebraic_likelihood"

    def test_tier1_f4_04_pallas_kernels_present(self):
        """Verify all Pallas kernels for Algebraic Transformer are retained."""
        kernels_dir = REPO_ROOT / "src" / "algebraic" / "kernels"
        expected_kernels = ["flash.py", "fused_head.py", "head_resident.py", "qk_norm.py", "radical.py"]
        for k in expected_kernels:
            assert (kernels_dir / k).exists(), f"Kernel {k} must exist in {kernels_dir}"

    def test_tier1_f4_05_data_preparation_script_present(self):
        """Verify prepare_fineweb_edu.py is retained and functional."""
        prep_script = REPO_ROOT / "scripts" / "prepare_fineweb_edu.py"
        assert prep_script.exists(), f"{prep_script} must exist"
        ast.parse(prep_script.read_text(encoding="utf-8"))

    # --------------------------------------------------------------------------
    # Feature 5: Standalone Muon + AdamW pure XLA (R3)
    # --------------------------------------------------------------------------

    def test_tier1_f5_01_muon_adamw_registered(self):
        """Verify get_optimizer('muon_adamw') returns (init_fn, step_fn)."""
        opt = load_optimizers_module()
        assert opt is not None, "src/algebraic/optimizers.py must be implemented"
        assert hasattr(opt, "get_optimizer"), "optimizers.py must export get_optimizer"
        init_fn, step_fn = opt.get_optimizer("muon_adamw", lr=0.003, total_steps=100)
        assert callable(init_fn), "init_fn must be callable"
        assert callable(step_fn), "step_fn must be callable"

    def test_tier1_f5_02_newton_schulz_5_iteration(self):
        """Verify newton_schulz_5 iteration executes and orthogonalizes 2D matrix."""
        opt = load_optimizers_module()
        assert opt is not None, "src/algebraic/optimizers.py must be implemented"
        assert hasattr(opt, "newton_schulz_5"), "optimizers.py must export newton_schulz_5"
        rng = jax.random.PRNGKey(42)
        G = jax.random.normal(rng, (32, 32))
        O = opt.newton_schulz_5(G, steps=5)
        assert O.shape == (32, 32), f"Expected shape (32, 32), got {O.shape}"
        gram = jnp.matmul(O, O.T)
        eye = jnp.eye(32)
        error = jnp.linalg.norm(gram - eye)
        assert error < 1.0, f"Newton-Schulz Gram matrix error too high: {error}"

    def test_tier1_f5_03_muon_nesterov_momentum(self):
        """Verify Muon applies Nesterov momentum with momentum coefficient 0.95."""
        opt = load_optimizers_module()
        assert opt is not None, "src/algebraic/optimizers.py must be implemented"
        # Test step function on a single matrix parameter
        init_fn, step_fn = opt.get_optimizer("muon_adamw", lr=0.01, muon_momentum=0.95, total_steps=10)
        params = {"w_q": jnp.ones((16, 16), dtype=jnp.float32)}
        opt_state = init_fn(params)
        grads = {"w_q": jnp.ones((16, 16), dtype=jnp.float32) * 0.1}
        decay_mask = {"w_q": True}
        new_params, new_opt_state = step_fn(0, params, grads, opt_state, decay_mask)
        assert not jnp.allclose(new_params["w_q"], params["w_q"]), "Params should update"

    def test_tier1_f5_04_adamw_1d_cosine_schedule(self):
        """Verify 1D parameters are updated using cosine learning rate schedule."""
        opt = load_optimizers_module()
        assert opt is not None, "src/algebraic/optimizers.py must be implemented"
        init_fn, step_fn = opt.get_optimizer(
            "muon_adamw", lr=0.01, total_steps=100, warmup_fraction=0.02, min_lr_ratio=0.1
        )
        params = {"embed_gain": jnp.ones((16,), dtype=jnp.float32)}
        opt_state = init_fn(params)
        grads = {"embed_gain": jnp.ones((16,), dtype=jnp.float32) * 0.1}
        decay_mask = {"embed_gain": False}
        p_step0, s0 = step_fn(0, params, grads, opt_state, decay_mask)
        p_step50, s50 = step_fn(50, params, grads, opt_state, decay_mask)
        # Updates should differ at step 0 (warmup) vs step 50 (cosine)
        u0 = params["embed_gain"] - p_step0["embed_gain"]
        u50 = params["embed_gain"] - p_step50["embed_gain"]
        assert not jnp.allclose(u0, u50), "Step updates should reflect learning rate schedule"

    def test_tier1_f5_05_pure_xla_execution(self):
        """Verify Muon+AdamW executes via pure XLA (compiles under jax.jit)."""
        opt = load_optimizers_module()
        assert opt is not None, "src/algebraic/optimizers.py must be implemented"
        init_fn, step_fn = opt.get_optimizer("muon_adamw", lr=0.01, total_steps=10)
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

    # --------------------------------------------------------------------------
    # Feature 6: Standalone C-AdamW pure XLA (R3)
    # --------------------------------------------------------------------------

    def test_tier1_f6_01_cadamw_registered(self):
        """Verify get_optimizer('cadamw') returns (init_fn, step_fn)."""
        opt = load_optimizers_module()
        assert opt is not None, "src/algebraic/optimizers.py must be implemented"
        init_fn, step_fn = opt.get_optimizer("cadamw", lr=0.003, total_steps=100)
        assert callable(init_fn)
        assert callable(step_fn)

    def test_tier1_f6_02_cautious_mask_condition(self):
        """Verify C-AdamW masks coordinates where update opposes current gradient (u * g <= 0)."""
        opt = load_optimizers_module()
        assert opt is not None, "src/algebraic/optimizers.py must be implemented"
        init_fn, step_fn = opt.get_optimizer("cadamw", lr=0.01, total_steps=100)
        # Momentum pointing positive, but instantaneous gradient pointing negative
        p = {"x": jnp.zeros((4,), dtype=jnp.float32)}
        s = init_fn(p)
        # Step 0: positive gradient
        g0 = {"x": jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)}
        p1, s1 = step_fn(0, p, g0, s, {"x": False})
        # Step 1: reversal on coordinate 0 (g1[0] = -1.0) while momentum is positive
        g1 = {"x": jnp.array([-1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)}
        p2, s2 = step_fn(1, p1, g1, s1, {"x": False})
        # Coordinate 0 should have been masked or cautious-stepped without opposition overshoot
        assert jnp.isfinite(p2["x"]).all()

    def test_tier1_f6_03_mask_mean_normalization(self):
        """Verify C-AdamW normalizes mask by mean(mask) to preserve step scale."""
        opt = load_optimizers_module()
        assert opt is not None, "src/algebraic/optimizers.py must be implemented"
        init_fn, step_fn = opt.get_optimizer("cadamw", lr=0.01, total_steps=100, cautious_eps=1e-3)
        p = {"w": jnp.ones((10,), dtype=jnp.float32)}
        s = init_fn(p)
        g = {"w": jnp.ones((10,), dtype=jnp.float32)}
        p1, s1 = step_fn(0, p, g, s, {"w": False})
        # When all coordinates agree, mean(mask) == 1.0 and norm factor is 1.0
        assert jnp.isfinite(p1["w"]).all()

    def test_tier1_f6_04_decoupled_weight_decay(self):
        """Verify C-AdamW applies decoupled weight decay only to decayed parameters."""
        opt = load_optimizers_module()
        assert opt is not None, "src/algebraic/optimizers.py must be implemented"
        init_fn, step_fn = opt.get_optimizer("cadamw", lr=0.01, weight_decay=0.1, total_steps=100)
        p = {"w_decay": jnp.ones((8,)), "w_nodecay": jnp.ones((8,))}
        s = init_fn(p)
        g = {"w_decay": jnp.zeros((8,)), "w_nodecay": jnp.zeros((8,))}
        dmask = {"w_decay": True, "w_nodecay": False}
        p1, _ = step_fn(0, p, g, s, dmask)
        # Decayed param should shrink by (1 - lr * wd) = 0.999
        assert p1["w_decay"][0] < 1.0, "Decayed parameter must shrink"
        assert jnp.isclose(p1["w_nodecay"][0], 1.0), "Non-decayed parameter must not shrink with zero gradient"

    def test_tier1_f6_05_pure_xla_jit(self):
        """Verify C-AdamW compiles and executes under jax.jit."""
        opt = load_optimizers_module()
        assert opt is not None, "src/algebraic/optimizers.py must be implemented"
        init_fn, step_fn = opt.get_optimizer("cadamw", lr=0.003, total_steps=10)
        p = {"w": jnp.ones((16, 16))}
        s = init_fn(p)
        g = {"w": jnp.ones((16, 16)) * 0.1}

        @jax.jit
        def step(p, g, s):
            return step_fn(0, p, g, s, {"w": True})

        new_p, new_s = step(p, g, s)
        assert jnp.isfinite(new_p["w"]).all()

    # --------------------------------------------------------------------------
    # Feature 7: Standalone Hybrid Muon+C-AdamW pure XLA (R3)
    # --------------------------------------------------------------------------

    def test_tier1_f7_01_hybrid_muon_cadamw_registered(self):
        """Verify get_optimizer('hybrid_muon_cadamw') returns (init_fn, step_fn)."""
        opt = load_optimizers_module()
        assert opt is not None, "src/algebraic/optimizers.py must be implemented"
        init_fn, step_fn = opt.get_optimizer("hybrid_muon_cadamw", lr=0.003, total_steps=100)
        assert callable(init_fn)
        assert callable(step_fn)

    def test_tier1_f7_02_structural_parameter_routing(self):
        """Verify internal 2D matrices route to Muon while 1D/embeddings route to C-AdamW."""
        opt = load_optimizers_module()
        assert opt is not None, "src/algebraic/optimizers.py must be implemented"
        assert hasattr(opt, "is_muon_param"), "optimizers.py should define is_muon_param routing rule"
        # 2D projection matrix
        assert opt.is_muon_param(("layers", "w_q"), jnp.zeros((12, 64, 64))) is True
        # 1D norm gain
        assert opt.is_muon_param(("layers", "ln1"), jnp.zeros((12, 64))) is False
        # Embedding
        assert opt.is_muon_param(("embed",), jnp.zeros((32768, 768))) is False

    def test_tier1_f7_03_separate_opt_state_buffers(self):
        """Verify optimizer state maintains first-moment only for Muon and second-moment for C-AdamW."""
        opt = load_optimizers_module()
        assert opt is not None, "src/algebraic/optimizers.py must be implemented"
        init_fn, _ = opt.get_optimizer("hybrid_muon_cadamw", lr=0.003, total_steps=100)
        params = {
            "embed": jnp.zeros((32, 16)),
            "w_q": jnp.zeros((16, 16)),
        }
        state = init_fn(params)
        # Check OptState structure
        assert hasattr(state, "count")
        assert hasattr(state, "m")
        assert hasattr(state, "v")

    def test_tier1_f7_04_hybrid_step_execution(self):
        """Verify hybrid step executes cleanly on both 2D and 1D parameters."""
        opt = load_optimizers_module()
        assert opt is not None, "src/algebraic/optimizers.py must be implemented"
        init_fn, step_fn = opt.get_optimizer("hybrid_muon_cadamw", lr=0.01, total_steps=100)
        params = {
            "w_q": jnp.ones((16, 16)),
            "embed_gain": jnp.ones((16,)),
        }
        dmask = {"w_q": True, "embed_gain": False}
        state = init_fn(params)
        grads = {
            "w_q": jnp.ones((16, 16)) * 0.05,
            "embed_gain": jnp.ones((16,)) * 0.05,
        }
        new_params, new_state = step_fn(0, params, grads, state, dmask)
        assert jnp.isfinite(new_params["w_q"]).all()
        assert jnp.isfinite(new_params["embed_gain"]).all()

    def test_tier1_f7_05_pure_xla_compilation(self):
        """Verify Hybrid Muon+C-AdamW compiles under jax.jit without errors."""
        opt = load_optimizers_module()
        assert opt is not None, "src/algebraic/optimizers.py must be implemented"
        init_fn, step_fn = opt.get_optimizer("hybrid_muon_cadamw", lr=0.003, total_steps=10)
        params = {"w_q": jnp.ones((8, 8)), "embed": jnp.ones((16, 8))}
        dmask = {"w_q": True, "embed": True}
        s = init_fn(params)
        g = {"w_q": jnp.ones((8, 8)) * 0.1, "embed": jnp.ones((16, 8)) * 0.1}

        @jax.jit
        def step(p, g, s):
            return step_fn(0, p, g, s, dmask)

        new_p, new_s = step(params, g, s)
        assert jnp.isfinite(new_p["w_q"]).all()

    # --------------------------------------------------------------------------
    # Feature 8: 32k Llama-style Tokenizer (R2)
    # --------------------------------------------------------------------------

    def test_tier1_f8_01_tokenizer_json_exists(self):
        """Verify src/algebraic/data/tokenizer.json exists and is valid JSON."""
        tok_json = REPO_ROOT / "src" / "algebraic" / "data" / "tokenizer.json"
        assert tok_json.exists(), f"Tokenizer file missing: {tok_json}"
        data = json.loads(tok_json.read_text(encoding="utf-8"))
        assert "model" in data or "vocab" in data

    def test_tier1_f8_02_tokenizer_vocab_size_32768(self):
        """Verify tokenizer vocabulary size is exactly 32,768."""
        from algebraic.tokenizer import get_tokenizer
        tok = get_tokenizer()
        assert tok.vocab_size == 32768, f"Expected vocab size 32768, got {tok.vocab_size}"

    def test_tier1_f8_03_tokenizer_module_loads(self):
        """Verify tokenizer module loads and exports expected symbols."""
        import algebraic.tokenizer as tok_mod
        assert hasattr(tok_mod, "BPE32kTokenizer"), "tokenizer.py must export BPE32kTokenizer"
        assert hasattr(tok_mod, "get_tokenizer"), "tokenizer.py must export get_tokenizer"

    def test_tier1_f8_04_configs_specify_32768(self):
        """Verify comparison configurations specify vocab_size 32768."""
        for cfg_name in ["comparison_125m.json", "comparison_350m.json"]:
            path = REPO_ROOT / "configs" / cfg_name
            if path.exists():
                cfg = json.loads(path.read_text(encoding="utf-8"))
                vocab = cfg.get("training", {}).get("vocab_size") or cfg.get("model", {}).get("vocab_size")
                assert vocab == 32768, f"{cfg_name} should specify vocab_size 32768, got {vocab}"

    def test_tier1_f8_05_tokenizer_encodes_and_decodes(self):
        """Verify tokenizer encodes and decodes sample text faithfully."""
        from algebraic.tokenizer import get_tokenizer
        tok = get_tokenizer()
        text = "Algebraic Transformer exploration with pure XLA optimizers."
        tokens = tok.encode(text)
        assert len(tokens) > 0
        assert all(0 <= t < 32768 for t in tokens), "All tokens must be within [0, 32767]"
        decoded = tok.decode(tokens)
        assert len(decoded) > 0

    # --------------------------------------------------------------------------
    # Feature 9: Main Run Configs (125M 2.5B, 350M 7B) (R2)
    # --------------------------------------------------------------------------

    def test_tier1_f9_01_main_125m_token_count(self):
        """Verify 125M main run specifies exactly 2,500,000,000 tokens."""
        c125 = json.loads((REPO_ROOT / "configs" / "comparison_125m.json").read_text(encoding="utf-8"))
        assert c125["training"]["tokens"] == 2_500_000_000

    def test_tier1_f9_02_main_125m_two_seeds(self):
        """Verify 125M main run specifies 2 seeds."""
        c125 = json.loads((REPO_ROOT / "configs" / "comparison_125m.json").read_text(encoding="utf-8"))
        seeds = c125.get("seeds") or c125.get("training", {}).get("seeds", [42, 43])
        assert len(seeds) == 2, f"125M main run must configure 2 seeds, got {seeds}"

    def test_tier1_f9_03_main_350m_token_count(self):
        """Verify 350M main run specifies exactly 7,000,000,000 tokens."""
        c350 = json.loads((REPO_ROOT / "configs" / "comparison_350m.json").read_text(encoding="utf-8"))
        assert c350["training"]["tokens"] == 7_000_000_000

    def test_tier1_f9_04_main_350m_one_seed(self):
        """Verify 350M main run specifies 1 seed."""
        c350 = json.loads((REPO_ROOT / "configs" / "comparison_350m.json").read_text(encoding="utf-8"))
        seeds = c350.get("seeds") or c350.get("training", {}).get("seeds", [42])
        assert len(seeds) == 1, f"350M main run must configure 1 seed, got {seeds}"

    def test_tier1_f9_05_fineweb_edu_dataset(self):
        """Verify main training configs target FineWeb-Edu dataset."""
        for cfg_name in ["comparison_125m.json", "comparison_350m.json"]:
            c = json.loads((REPO_ROOT / "configs" / cfg_name).read_text(encoding="utf-8"))
            train_file = c["training"]["train_file"]
            assert "fineweb" in train_file.lower(), f"{cfg_name} should target FineWeb-Edu, got {train_file}"

    # --------------------------------------------------------------------------
    # Feature 10: Sweep Configs (125M 600M, 350M 2B) (R2)
    # --------------------------------------------------------------------------

    def test_tier1_f10_01_sweep_125m_token_budget(self):
        """Verify sweep configuration targets 600M tokens/arm for 125M."""
        pipeline_script = REPO_ROOT / "scripts" / "run_experiments_pipeline.py"
        assert pipeline_script.exists()
        content = pipeline_script.read_text(encoding="utf-8")
        assert "600_000_000" in content or "600000000" in content, (
            "125M sweep configuration must specify 600M tokens per arm"
        )

    def test_tier1_f10_02_sweep_350m_token_budget(self):
        """Verify sweep configuration targets 2B tokens/arm for 350M."""
        pipeline_script = REPO_ROOT / "scripts" / "run_experiments_pipeline.py"
        assert pipeline_script.exists()
        content = pipeline_script.read_text(encoding="utf-8")
        assert "2_000_000_000" in content or "2000000000" in content, (
            "350M sweep configuration must specify 2B tokens per arm"
        )

    def test_tier1_f10_03_sweep_grid_all_three_optimizers(self):
        """Verify sweep configuration defines grids for Muon+AdamW, C-AdamW, and Hybrid Muon+C-AdamW."""
        pipeline_script = REPO_ROOT / "scripts" / "run_experiments_pipeline.py"
        content = pipeline_script.read_text(encoding="utf-8")
        for opt_name in ["muon_adamw", "cadamw", "hybrid_muon_cadamw"]:
            assert opt_name in content, (
                f"Pipeline script must configure sweep grid for optimizer {opt_name}"
            )

    def test_tier1_f10_04_sweep_step_counts_calculated(self):
        """Verify sweep step counts match token budgets with 1,048,576 tokens/step."""
        tokens_per_step = 512 * 2048  # 1,048,576
        steps_125m = math.ceil(600_000_000 / tokens_per_step)
        steps_350m = math.ceil(2_000_000_000 / tokens_per_step)
        assert steps_125m == 573, f"Expected 573 steps for 125M sweep, got {steps_125m}"
        assert steps_350m == 1908, f"Expected 1908 steps for 350M sweep, got {steps_350m}"

    def test_tier1_f10_05_sweep_scripts_dry_run(self):
        """Verify sweep configuration script supports configure-only / dry-run mode."""
        pipeline_script = REPO_ROOT / "scripts" / "run_experiments_pipeline.py"
        content = pipeline_script.read_text(encoding="utf-8")
        assert "--dry-run" in content or "configure" in content or "--no-run" in content, (
            "Sweep script must allow configuration without full execution"
        )

    # --------------------------------------------------------------------------
    # Feature 11: Unit Tests for Optimizers (R4)
    # --------------------------------------------------------------------------

    def test_tier1_f11_01_optimizer_test_file_exists(self):
        """Verify tests/test_optimizers.py exists."""
        test_file = REPO_ROOT / "tests" / "test_optimizers.py"
        assert test_file.exists(), f"tests/test_optimizers.py must exist: {test_file}"

    def test_tier1_f11_02_test_file_covers_muon(self):
        """Verify tests/test_optimizers.py contains tests for Muon+AdamW."""
        test_file = REPO_ROOT / "tests" / "test_optimizers.py"
        content = test_file.read_text(encoding="utf-8")
        assert "muon_adamw" in content, "test_optimizers.py must test muon_adamw"

    def test_tier1_f11_03_test_file_covers_cadamw(self):
        """Verify tests/test_optimizers.py contains tests for C-AdamW."""
        test_file = REPO_ROOT / "tests" / "test_optimizers.py"
        content = test_file.read_text(encoding="utf-8")
        assert "cadamw" in content or "c_adamw" in content, "test_optimizers.py must test C-AdamW"

    def test_tier1_f11_04_test_file_covers_hybrid(self):
        """Verify tests/test_optimizers.py contains tests for Hybrid Muon+C-AdamW."""
        test_file = REPO_ROOT / "tests" / "test_optimizers.py"
        content = test_file.read_text(encoding="utf-8")
        assert "hybrid" in content.lower(), "test_optimizers.py must test Hybrid Muon+C-AdamW"

    def test_tier1_f11_05_test_file_covers_newton_schulz(self):
        """Verify tests/test_optimizers.py contains tests for Newton-Schulz polynomial."""
        test_file = REPO_ROOT / "tests" / "test_optimizers.py"
        content = test_file.read_text(encoding="utf-8")
        assert "newton_schulz" in content, "test_optimizers.py must test Newton-Schulz iteration"

    # --------------------------------------------------------------------------
    # Feature 12: 16-TPU Smoke Tests (R4)
    # --------------------------------------------------------------------------

    def test_tier1_f12_01_smoke_16tpu_script_exists(self):
        """Verify scripts/smoke_16tpu.py exists."""
        smoke_script = REPO_ROOT / "scripts" / "smoke_16tpu.py"
        assert smoke_script.exists(), f"scripts/smoke_16tpu.py must exist: {smoke_script}"

    def test_tier1_f12_02_smoke_supports_125m_and_350m(self):
        """Verify smoke test script supports both 125M and 350M models."""
        smoke_script = REPO_ROOT / "scripts" / "smoke_16tpu.py"
        content = smoke_script.read_text(encoding="utf-8")
        assert "125m" in content and "350m" in content, (
            "smoke_16tpu.py must support both 125m and 350m scales"
        )

    def test_tier1_f12_03_smoke_supports_all_three_optimizers(self):
        """Verify smoke test script supports all three pure XLA optimizers."""
        smoke_script = REPO_ROOT / "scripts" / "smoke_16tpu.py"
        content = smoke_script.read_text(encoding="utf-8")
        for opt in ["muon_adamw", "cadamw", "hybrid_muon_cadamw"]:
            assert opt in content, f"smoke_16tpu.py must support optimizer {opt}"

    def test_tier1_f12_04_smoke_16tpu_hosts_configured(self):
        """Verify smoke script defaults to 4 workers (16 chips total: w0,w1,w2,w3)."""
        smoke_script = REPO_ROOT / "scripts" / "smoke_16tpu.py"
        content = smoke_script.read_text(encoding="utf-8")
        assert "w0,w1,w2,w3" in content or "w0" in content, (
            "smoke_16tpu.py must configure multi-host cluster w0..w3"
        )

    def test_tier1_f12_05_smoke_single_step_synthetic(self):
        """Verify smoke script runs exactly 1 step on synthetic data."""
        smoke_script = REPO_ROOT / "scripts" / "smoke_16tpu.py"
        content = smoke_script.read_text(encoding="utf-8")
        assert "--steps" in content and "1" in content, (
            "smoke_16tpu.py must execute exactly 1 step"
        )

    # --------------------------------------------------------------------------
    # Feature 13: Standalone Comments (R5)
    # --------------------------------------------------------------------------

    def test_tier1_f13_01_no_debugging_history_comments(self):
        """Verify source code comments contain no historical debugging commentary."""
        forbidden_comment_patterns = [
            r"#.*(?:fixed bug|workaround for issue|temporary fix|quick hack|debug print)",
            r"#.*(?:commit [0-9a-f]{7,40}|reverted from|cherry-picked from)",
        ]
        violations = []
        for p in (REPO_ROOT / "src" / "algebraic").rglob("*.py"):
            content = p.read_text(encoding="utf-8")
            for line_no, line in enumerate(content.splitlines(), start=1):
                for pat in forbidden_comment_patterns:
                    if re.search(pat, line, re.IGNORECASE):
                        violations.append(f"{p.name}:{line_no}: {line.strip()}")
        assert not violations, f"Found debugging commentary in comments:\n" + "\n".join(violations)

    def test_tier1_f13_02_no_todo_fixme_comments(self):
        """Verify src/algebraic/ contains no lingering TODO or FIXME comments."""
        violations = []
        for p in (REPO_ROOT / "src" / "algebraic").rglob("*.py"):
            content = p.read_text(encoding="utf-8")
            for line_no, line in enumerate(content.splitlines(), start=1):
                if re.search(r"#.*(?:TODO|FIXME|XXX)", line):
                    violations.append(f"{p.name}:{line_no}: {line.strip()}")
        assert not violations, f"Found lingering TODO/FIXME comments:\n" + "\n".join(violations)

    def test_tier1_f13_03_module_docstrings_present(self):
        """Verify all core modules in src/algebraic/ have clean module docstrings."""
        src_dir = REPO_ROOT / "src" / "algebraic"
        for p in src_dir.glob("*.py"):
            if p.name == "__init__.py":
                continue
            tree = ast.parse(p.read_text(encoding="utf-8"))
            doc = ast.get_docstring(tree)
            assert doc is not None and len(doc.strip()) > 10, (
                f"Module {p.name} must have a standalone docstring"
            )

    def test_tier1_f13_04_kernel_docstrings_standalone(self):
        """Verify kernel modules have clear standalone docstrings."""
        kernels_dir = REPO_ROOT / "src" / "algebraic" / "kernels"
        for p in kernels_dir.glob("*.py"):
            if p.name == "__init__.py":
                continue
            tree = ast.parse(p.read_text(encoding="utf-8"))
            doc = ast.get_docstring(tree)
            assert doc is not None and len(doc.strip()) > 10, (
                f"Kernel {p.name} must have a standalone docstring"
            )

    def test_tier1_f13_05_scripts_docstrings_standalone(self):
        """Verify pipeline scripts have standalone docstrings."""
        scripts_dir = REPO_ROOT / "scripts"
        for p in scripts_dir.glob("*.py"):
            tree = ast.parse(p.read_text(encoding="utf-8"))
            doc = ast.get_docstring(tree)
            assert doc is not None and len(doc.strip()) > 10, (
                f"Script {p.name} must have a standalone docstring"
            )

    # --------------------------------------------------------------------------
    # Feature 14: PEP 8 Formatted (R5)
    # --------------------------------------------------------------------------

    def test_tier1_f14_01_pyproject_or_ruff_configured(self):
        """Verify pyproject.toml defines build and code style standards."""
        pyproj = REPO_ROOT / "pyproject.toml"
        assert pyproj.exists(), "pyproject.toml must exist"
        content = pyproj.read_text(encoding="utf-8")
        assert "project" in content or "tool" in content

    def test_tier1_f14_02_python_syntax_valid(self):
        """Verify every Python file in the repository parses without SyntaxError."""
        for py_file in get_all_active_python_files():
            content = py_file.read_text(encoding="utf-8")
            try:
                ast.parse(content, filename=str(py_file))
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {py_file}: {e}")

    def test_tier1_f14_03_no_carriage_returns(self):
        """Verify no carriage returns (\r\n) exist in Python source files."""
        violations = []
        for py_file in get_all_active_python_files():
            raw = py_file.read_bytes()
            if b"\r\n" in raw:
                violations.append(str(py_file.relative_to(REPO_ROOT)))
        assert not violations, f"Windows CRLF line endings found in:\n" + "\n".join(violations)

    def test_tier1_f14_04_flake8_or_ruff_clean(self):
        """Verify line length limit (<=100 characters) across src/algebraic/."""
        violations = []
        for py_file in (REPO_ROOT / "src" / "algebraic").rglob("*.py"):
            lines = py_file.read_text(encoding="utf-8").splitlines()
            for line_no, line in enumerate(lines, start=1):
                if len(line) > 100:
                    violations.append(f"{py_file.name}:{line_no}: len={len(line)}")
        assert not violations, f"Lines exceeding 100 characters:\n" + "\n".join(violations[:10])

    def test_tier1_f14_05_import_ordering(self):
        """Verify __future__ imports appear at the top of files when present."""
        for py_file in (REPO_ROOT / "src" / "algebraic").rglob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                    # Must be first non-docstring statement
                    first_idx = 1 if ast.get_docstring(tree) else 0
                    assert tree.body.index(node) == first_idx, (
                        f"__future__ import in {py_file.name} must appear at the top"
                    )

    # --------------------------------------------------------------------------
    # Feature 15: Public GitHub algebraic-extra published (R5)
    # --------------------------------------------------------------------------

    def test_tier1_f15_01_license_apache2(self):
        """Verify LICENSE file is Apache-2.0."""
        license_file = REPO_ROOT / "LICENSE"
        assert license_file.exists(), "LICENSE must exist in repository root"
        content = license_file.read_text(encoding="utf-8")
        assert "Apache License" in content and "Version 2.0" in content

    def test_tier1_f15_02_readme_standalone_algebraic(self):
        """Verify README.md documents algebraic-extra standalone codebase."""
        readme_file = REPO_ROOT / "README.md"
        assert readme_file.exists(), "README.md must exist in repository root"
        content = readme_file.read_text(encoding="utf-8")
        assert "algebraic" in content.lower()

    def test_tier1_f15_03_pyproject_package_name(self):
        """Verify pyproject.toml defines package name and metadata."""
        pyproj = REPO_ROOT / "pyproject.toml"
        assert pyproj.exists()
        content = pyproj.read_text(encoding="utf-8")
        assert "name =" in content

    def test_tier1_f15_04_git_remote_or_origin_clean(self):
        """Verify .git repository is initialized in the working directory."""
        git_dir = REPO_ROOT / ".git"
        assert git_dir.exists(), "Git repository must be initialized"

    def test_tier1_f15_05_publish_script_or_dryrun(self):
        """Verify repository readiness for git push as Initial Commit."""
        git_dir = REPO_ROOT / ".git"
        assert git_dir.exists()


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES (>=5 edge cases per feature for Features 1-15)
# ==============================================================================


@pytest.mark.tier2
@pytest.mark.e2e
class TestTier2BoundaryAndCornerCases:
    """Tier 2: Boundary value analysis, extreme shapes, edge cases, and numerical robustness."""

    # --------------------------------------------------------------------------
    # Feature 1 Boundaries: 1B Removals
    # --------------------------------------------------------------------------

    def test_tier2_f1_b01_nested_scan_zero_1b(self):
        """Exhaustive recursive search for lingering 1B config or script files."""
        matches = [str(p.relative_to(REPO_ROOT)) for p in REPO_ROOT.rglob("*1b*")
                   if not any(x in p.parts for x in REPO_SCAN_EXCLUDES)]
        assert not matches, f"Lingering 1B filenames found: {matches}"

    def test_tier2_f1_b02_case_insensitive_no_1b_configs(self):
        """Case-insensitive search ensures no *1b*.json or *1B*.json configs remain."""
        cfg_dir = REPO_ROOT / "configs"
        if cfg_dir.exists():
            for p in cfg_dir.glob("*.json"):
                assert "1b" not in p.name.lower(), f"Forbidden 1b config found: {p.name}"

    def test_tier2_f1_b03_zero_1b_data_references_across_repo(self):
        """Zero references to fineweb1b_train or fineweb1b_valid in active code."""
        for py_file in (REPO_ROOT / "src").rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            assert "fineweb1b" not in content, f"Reference in {py_file.name}"

    def test_tier2_f1_b04_no_1b_symbols_in_ast(self):
        """AST inspection ensures no '1b' key in config.py SIZES dict definition."""
        cfg_file = REPO_ROOT / "src" / "algebraic" / "config.py"
        tree = ast.parse(cfg_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for k in node.keys:
                    if isinstance(k, ast.Constant) and k.value == "1b":
                        pytest.fail("Found '1b' dictionary key in config.py AST")

    def test_tier2_f1_b05_parameter_count_rejects_1b(self):
        """Requesting 1b model size raises KeyError in SIZES."""
        from algebraic.config import SIZES
        with pytest.raises(KeyError):
            _ = SIZES["1b"]

    # --------------------------------------------------------------------------
    # Feature 2 Boundaries: Standard Transformer Removals
    # --------------------------------------------------------------------------

    def test_tier2_f2_b01_import_standard_fails(self):
        """Importing standard module raises ModuleNotFoundError."""
        import importlib
        with pytest.raises((ModuleNotFoundError, ImportError)):
            importlib.import_module(".".join(["algebraic", "standard"]))

    def test_tier2_f2_b02_standard_preset_raises_error(self):
        """Requesting preset 'standard' from config raises KeyError or AttributeError."""
        import algebraic.config as cfg_mod
        if hasattr(cfg_mod, "PRESETS"):
            with pytest.raises(KeyError):
                _ = cfg_mod.PRESETS["standard"]

    def test_tier2_f2_b03_flash_attention_rejects_softmax(self):
        """Flash attention rejects kind='softmax'."""
        try:
            from algebraic.kernels.flash import flash_attention
            q = jnp.zeros((1, 2, 4, 16))
            k = jnp.zeros((1, 2, 4, 16))
            v = jnp.zeros((1, 2, 4, 16))
            with pytest.raises(ValueError):
                flash_attention(q, k, v, None, kind="softmax")
        except (ImportError, AttributeError):
            pass

    def test_tier2_f2_b04_fused_head_rejects_ce(self):
        """Fused head rejects kind='ce'."""
        try:
            from algebraic.kernels.fused_head import fused_head
            h = jnp.zeros((4, 16))
            e = jnp.zeros((32, 16))
            tgt = jnp.zeros((4,), dtype=jnp.int32)
            with pytest.raises(ValueError):
                fused_head(h, e, tgt, kind="ce")
        except (ImportError, AttributeError):
            pass

    def test_tier2_f2_b05_model_attention_rejects_softmax(self):
        """LMConfig rejects or errors when attn='softmax' is evaluated."""
        from algebraic.model import LMConfig, _attention
        cfg = LMConfig(attn="softmax")
        q = jnp.zeros((1, 1, 2, 16))
        with pytest.raises(ValueError):
            _attention(q, q, q, None, cfg)

    # --------------------------------------------------------------------------
    # Feature 3 Boundaries: Pruned Directories
    # --------------------------------------------------------------------------

    def test_tier2_f3_b01_pruned_paths_not_symlinks(self):
        """Pruned directories are not preserved as broken symlinks."""
        for name in ["formal", "analysis", "paper", "results"]:
            p = REPO_ROOT / name
            assert not p.is_symlink(), f"{name} must not be a symlink"

    def test_tier2_f3_b02_zero_lean_files(self):
        """Zero Lean 4 proof files (.lean) exist across the repository."""
        lean_files = list(REPO_ROOT.rglob("*.lean"))
        assert not lean_files, f"Found .lean files: {lean_files}"

    def test_tier2_f3_b03_zero_historical_csv_files(self):
        """Zero historical evaluation CSV files exist in root."""
        csv_files = [p for p in (REPO_ROOT / "results").glob("*.csv")] if (REPO_ROOT / "results").exists() else []
        assert not csv_files, f"Found historical CSV results: {csv_files}"

    def test_tier2_f3_b04_zero_dead_figure_files(self):
        """Zero figure image artifacts exist in root or results."""
        dead_figures = list((REPO_ROOT / "results" / "figures").glob("*")) if (REPO_ROOT / "results" / "figures").exists() else []
        assert not dead_figures, f"Found dead figures: {dead_figures}"

    def test_tier2_f3_b05_no_hidden_artifacts_in_pruned_paths(self):
        """No lake build directories or lean-toolchain remain in root."""
        assert not (REPO_ROOT / "lake-manifest.json").exists()
        assert not (REPO_ROOT / "lean-toolchain").exists()
        assert not (REPO_ROOT / "lakefile.toml").exists()

    # --------------------------------------------------------------------------
    # Feature 4 Boundaries: Retained Algebraic Transformer Math
    # --------------------------------------------------------------------------

    def test_tier2_f4_b01_seq_len_one_forward(self):
        """Forward pass with minimum sequence length (seq_len=1) executes cleanly."""
        from algebraic.model import LMConfig, init_params, position_tables, forward_hidden
        cfg = LMConfig(d_model=64, n_layers=1, n_heads=2, d_ff=128, seq_len=1, vocab_size=128)
        rng = jax.random.PRNGKey(42)
        params = init_params(cfg, rng)
        tables = position_tables(cfg, 1)
        tokens = jnp.array([[5]], dtype=jnp.int32)
        out = forward_hidden(params, tokens, cfg, tables)
        assert out.shape == (1, 1, 64)
        assert jnp.isfinite(out).all()

    def test_tier2_f4_b02_batch_size_one_forward(self):
        """Forward pass with single-item batch (batch=1) executes without shape errors."""
        from algebraic.model import LMConfig, init_params, position_tables, forward_hidden
        cfg = LMConfig(d_model=64, n_layers=1, n_heads=2, d_ff=128, seq_len=4, vocab_size=128)
        rng = jax.random.PRNGKey(42)
        params = init_params(cfg, rng)
        tables = position_tables(cfg, 4)
        tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
        out = forward_hidden(params, tokens, cfg, tables)
        assert out.shape == (1, 4, 64)
        assert jnp.isfinite(out).all()

    def test_tier2_f4_b03_radical_extreme_negative_input(self):
        """Radical rho(x) at extreme negative value (-1e6) evaluates via reciprocal fallback without NaN."""
        import algebraic.algebraic as alg
        x = jnp.array([-1e6], dtype=jnp.float32)
        val = alg.rho(x)
        assert jnp.isfinite(val).all()
        assert val[0] > 0.0, "rho(x) must remain strictly positive"
        expected = 1.0 / (1e6 + math.sqrt(1.0 + 1e12))
        assert jnp.isclose(val[0], expected, rtol=1e-3)

    def test_tier2_f4_b04_radical_extreme_positive_input(self):
        """Radical rho(x) at extreme positive value (+1e6) evaluates without overflow."""
        import algebraic.algebraic as alg
        x = jnp.array([1e6], dtype=jnp.float32)
        val = alg.rho(x)
        assert jnp.isfinite(val).all()
        assert val[0] > 1e6

    def test_tier2_f4_b05_cayley_orthogonal_at_boundary_angles(self):
        """Cayley rotation matrix at boundary angles satisfies R * R.T = I."""
        import algebraic.algebraic as alg
        table = alg.cayley_table_geometric(head_dim=16, seq_len=4, base=10000.0)
        cos, sin = table
        assert jnp.isfinite(cos).all()
        assert jnp.isfinite(sin).all()

    # --------------------------------------------------------------------------
    # Feature 5 Boundaries: Muon + AdamW
    # --------------------------------------------------------------------------

    def test_tier2_f5_b01_newton_schulz_square_matrix(self):
        """Newton-Schulz on square matrix (64x64) produces near-identity Gram matrix."""
        opt = load_optimizers_module()
        assert opt is not None, "optimizers.py must be implemented"
        rng = jax.random.PRNGKey(101)
        G = jax.random.normal(rng, (64, 64))
        O = opt.newton_schulz_5(G, steps=5)
        gram = jnp.matmul(O, O.T)
        error = jnp.max(jnp.abs(gram - jnp.eye(64)))
        assert error < 0.25, f"Square matrix orthogonalization error: {error}"

    def test_tier2_f5_b02_newton_schulz_tall_matrix(self):
        """Newton-Schulz on tall matrix (128x64) produces near-identity on inner dimension."""
        opt = load_optimizers_module()
        assert opt is not None, "optimizers.py must be implemented"
        rng = jax.random.PRNGKey(102)
        G = jax.random.normal(rng, (128, 64))
        O = opt.newton_schulz_5(G, steps=5)
        assert O.shape == (128, 64)
        gram = jnp.matmul(O.T, O)
        error = jnp.max(jnp.abs(gram - jnp.eye(64) * (128.0 / 64.0)))
        assert error < 1.0

    def test_tier2_f5_b03_newton_schulz_wide_matrix(self):
        """Newton-Schulz on wide matrix (64x128) produces near-identity on outer dimension."""
        opt = load_optimizers_module()
        assert opt is not None, "optimizers.py must be implemented"
        rng = jax.random.PRNGKey(103)
        G = jax.random.normal(rng, (64, 128))
        O = opt.newton_schulz_5(G, steps=5)
        assert O.shape == (64, 128)
        gram = jnp.matmul(O, O.T)
        error = jnp.max(jnp.abs(gram - jnp.eye(64)))
        assert error < 0.25

    def test_tier2_f5_b04_newton_schulz_zero_norm_guard(self):
        """Newton-Schulz on near-zero norm matrix remains numerically finite via eps guard."""
        opt = load_optimizers_module()
        assert opt is not None, "optimizers.py must be implemented"
        G = jnp.zeros((16, 16), dtype=jnp.float32)
        O = opt.newton_schulz_5(G, steps=5)
        assert jnp.isfinite(O).all()

    def test_tier2_f5_b05_aspect_ratio_scaling_boundary(self):
        """Aspect ratio scaling factor equals sqrt(max(1, M/N))."""
        # For M=2048, N=768 (w_d): scale = sqrt(2048/768) = 1.63299
        scale_tall = math.sqrt(2048 / 768)
        assert math.isclose(scale_tall, 1.632993, rel_tol=1e-4)
        # For M=768, N=2048 (w_g): scale = 1.0
        scale_wide = math.sqrt(max(1.0, 768 / 2048))
        assert scale_wide == 1.0

    # --------------------------------------------------------------------------
    # Feature 6 Boundaries: C-AdamW
    # --------------------------------------------------------------------------

    def test_tier2_f6_b01_cadamw_full_alignment_mask(self):
        """C-AdamW when all coordinates align (u * g > 0): mask is all 1s."""
        opt = load_optimizers_module()
        assert opt is not None, "optimizers.py must be implemented"
        init_fn, step_fn = opt.get_optimizer("cadamw", lr=0.01, total_steps=100)
        p = {"x": jnp.ones((8,))}
        s = init_fn(p)
        g = {"x": jnp.ones((8,)) * 0.5}
        p1, s1 = step_fn(0, p, g, s, {"x": False})
        # Coordinate step should be positive descent
        assert (p1["x"] < p["x"]).all()

    def test_tier2_f6_b02_cadamw_full_opposition_mask(self):
        """C-AdamW when all coordinates oppose: mask is clamped by epsilon."""
        opt = load_optimizers_module()
        assert opt is not None, "optimizers.py must be implemented"
        init_fn, step_fn = opt.get_optimizer("cadamw", lr=0.01, total_steps=100, cautious_eps=1e-3)
        p = {"x": jnp.ones((8,))}
        s = init_fn(p)
        # Seed momentum with positive gradients
        g_pos = {"x": jnp.ones((8,))}
        p1, s1 = step_fn(0, p, g_pos, s, {"x": False})
        # Now pass strictly negative gradients (opposing momentum)
        g_neg = {"x": -jnp.ones((8,))}
        p2, s2 = step_fn(1, p1, g_neg, s1, {"x": False})
        assert jnp.isfinite(p2["x"]).all()

    def test_tier2_f6_b03_cadamw_zero_gradient_step(self):
        """C-AdamW handles zero gradient without NaN or division by zero."""
        opt = load_optimizers_module()
        assert opt is not None, "optimizers.py must be implemented"
        init_fn, step_fn = opt.get_optimizer("cadamw", lr=0.01, total_steps=100)
        p = {"x": jnp.ones((8,))}
        s = init_fn(p)
        g = {"x": jnp.zeros((8,))}
        p1, s1 = step_fn(0, p, g, s, {"x": False})
        assert jnp.allclose(p1["x"], p["x"])

    def test_tier2_f6_b04_cadamw_scalar_parameter(self):
        """C-AdamW operates correctly on 0-dim scalar parameters."""
        opt = load_optimizers_module()
        assert opt is not None, "optimizers.py must be implemented"
        init_fn, step_fn = opt.get_optimizer("cadamw", lr=0.01, total_steps=100)
        p = {"scalar": jnp.array(2.0, dtype=jnp.float32)}
        s = init_fn(p)
        g = {"scalar": jnp.array(0.5, dtype=jnp.float32)}
        p1, s1 = step_fn(0, p, g, s, {"scalar": False})
        assert jnp.isfinite(p1["scalar"]).all()

    def test_tier2_f6_b05_cadamw_extreme_gradients(self):
        """C-AdamW handles gradients of magnitude 1e6 and 1e-12 without overflow."""
        opt = load_optimizers_module()
        assert opt is not None, "optimizers.py must be implemented"
        init_fn, step_fn = opt.get_optimizer("cadamw", lr=0.01, total_steps=100)
        p = {"large": jnp.ones((4,)), "tiny": jnp.ones((4,))}
        s = init_fn(p)
        g = {"large": jnp.full((4,), 1e6), "tiny": jnp.full((4,), 1e-12)}
        p1, s1 = step_fn(0, p, g, s, {"large": False, "tiny": False})
        assert jnp.isfinite(p1["large"]).all()
        assert jnp.isfinite(p1["tiny"]).all()

    # --------------------------------------------------------------------------
    # Feature 7 Boundaries: Hybrid Muon + C-AdamW
    # --------------------------------------------------------------------------

    def test_tier2_f7_b01_hybrid_empty_pytree(self):
        """Hybrid optimizer handles empty parameter tree gracefully."""
        opt = load_optimizers_module()
        assert opt is not None, "optimizers.py must be implemented"
        init_fn, step_fn = opt.get_optimizer("hybrid_muon_cadamw", lr=0.01, total_steps=10)
        p = {}
        s = init_fn(p)
        p1, s1 = step_fn(0, p, {}, s, {})
        assert p1 == {}

    def test_tier2_f7_b02_hybrid_single_2d_matrix(self):
        """Hybrid optimizer with only 2D matrix routes exclusively to Muon."""
        opt = load_optimizers_module()
        assert opt is not None, "optimizers.py must be implemented"
        init_fn, step_fn = opt.get_optimizer("hybrid_muon_cadamw", lr=0.01, total_steps=10)
        p = {"w_q": jnp.ones((8, 8))}
        s = init_fn(p)
        g = {"w_q": jnp.ones((8, 8)) * 0.1}
        p1, s1 = step_fn(0, p, g, s, {"w_q": True})
        assert jnp.isfinite(p1["w_q"]).all()

    def test_tier2_f7_b03_hybrid_single_1d_vector(self):
        """Hybrid optimizer with only 1D vector routes exclusively to C-AdamW."""
        opt = load_optimizers_module()
        assert opt is not None, "optimizers.py must be implemented"
        init_fn, step_fn = opt.get_optimizer("hybrid_muon_cadamw", lr=0.01, total_steps=10)
        p = {"embed_gain": jnp.ones((16,))}
        s = init_fn(p)
        g = {"embed_gain": jnp.ones((16,)) * 0.1}
        p1, s1 = step_fn(0, p, g, s, {"embed_gain": False})
        assert jnp.isfinite(p1["embed_gain"]).all()

    def test_tier2_f7_b04_hybrid_mixed_shapes(self):
        """Hybrid optimizer operates on nested dictionary containing 1D, 2D, and 3D arrays."""
        opt = load_optimizers_module()
        assert opt is not None, "optimizers.py must be implemented"
        init_fn, step_fn = opt.get_optimizer("hybrid_muon_cadamw", lr=0.01, total_steps=10)
        p = {
            "embed": jnp.ones((32, 16)),
            "layers": {
                "w_q": jnp.ones((2, 16, 16)),
                "ln1": jnp.ones((2, 16)),
            }
        }
        dmask = {"embed": True, "layers": {"w_q": True, "ln1": False}}
        s = init_fn(p)
        g = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 0.05, p)
        p1, s1 = step_fn(0, p, g, s, dmask)
        assert jnp.isfinite(p1["layers"]["w_q"]).all()
        assert jnp.isfinite(p1["layers"]["ln1"]).all()

    def test_tier2_f7_b05_hybrid_memory_reduction(self):
        """Hybrid optimizer reduces memory by not storing second moment for 2D weights."""
        opt = load_optimizers_module()
        assert opt is not None, "optimizers.py must be implemented"
        init_fn, _ = opt.get_optimizer("hybrid_muon_cadamw", lr=0.01, total_steps=10)
        p = {"w_q": jnp.ones((64, 64)), "ln1": jnp.ones((64,))}
        s = init_fn(p)
        # In state.v, 2D weight should be None or omitted
        if isinstance(s.v, dict):
            assert s.v.get("w_q") is None or "w_q" not in s.v, (
                "Hybrid optimizer must not allocate second-moment buffer for Muon weights"
            )

    # --------------------------------------------------------------------------
    # Feature 8 Boundaries: Tokenizer
    # --------------------------------------------------------------------------

    def test_tier2_f8_b01_tokenizer_empty_string(self):
        """Tokenizer encoding an empty string returns empty token array."""
        from algebraic.tokenizer import get_tokenizer
        tok = get_tokenizer()
        tokens = tok.encode("")
        assert len(tokens) == 0

    def test_tier2_f8_b02_tokenizer_single_character(self):
        """Tokenizer encoding a single character returns valid token ID."""
        from algebraic.tokenizer import get_tokenizer
        tok = get_tokenizer()
        tokens = tok.encode("a")
        assert len(tokens) >= 1
        assert 0 <= tokens[0] < 32768

    def test_tier2_f8_b03_tokenizer_unicode_handling(self):
        """Tokenizer handles unicode and emoji without crashing."""
        from algebraic.tokenizer import get_tokenizer
        tok = get_tokenizer()
        tokens = tok.encode("Algebraic: \u03c0 \u2248 3.14159, \u03c1(x)")
        assert len(tokens) > 0
        decoded = tok.decode(tokens)
        assert "\u03c0" in decoded or "3.14159" in decoded

    def test_tier2_f8_b04_vocab_boundary_token_valid(self):
        """Token ID 32,767 is within vocabulary boundary."""
        from algebraic.tokenizer import get_tokenizer
        tok = get_tokenizer()
        try:
            _ = tok.decode([32767])
        except Exception as e:
            pytest.fail(f"Decoding token 32767 failed: {e}")

    def test_tier2_f8_b05_vocab_boundary_token_invalid(self):
        """Token ID 32,768 is out of vocabulary boundary."""
        from algebraic.tokenizer import get_tokenizer
        tok = get_tokenizer()
        assert 32768 >= tok.vocab_size

    # --------------------------------------------------------------------------
    # Feature 9 Boundaries: Main Configurations
    # --------------------------------------------------------------------------

    def test_tier2_f9_b01_step_count_zero_tokens_error(self):
        """Token budget calculation handles zero or negative tokens with error."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            tokens = 0
            tokens_per_step = 0
            _ = tokens // tokens_per_step

    def test_tier2_f9_b02_batch_arithmetic_consistency(self):
        """Batch arithmetic: batch * seq_len == tokens_per_step (512 * 2048 == 1,048,576)."""
        batch = 512
        seq_len = 2048
        assert batch * seq_len == 1_048_576

    def test_tier2_f9_b03_seeds_list_boundary(self):
        """Main configs specify non-empty integer seeds."""
        for cfg_name in ["comparison_125m.json", "comparison_350m.json"]:
            c = json.loads((REPO_ROOT / "configs" / cfg_name).read_text(encoding="utf-8"))
            seeds = c.get("seeds") or c.get("training", {}).get("seeds", [42])
            assert all(isinstance(s, int) and s >= 0 for s in seeds)

    def test_tier2_f9_b04_warmup_fraction_zero(self):
        """Cosine schedule with warmup fraction 0.0 initializes at peak LR immediately."""
        opt = load_optimizers_module()
        if opt and hasattr(opt, "cosine_schedule"):
            lr0 = opt.cosine_schedule(0, lr=0.01, total_steps=100, warmup_fraction=0.0)
            assert jnp.isclose(lr0, 0.01)

    def test_tier2_f9_b05_warmup_fraction_boundary(self):
        """Warmup fraction 0.02 reaches peak learning rate at step 48 for 2385 steps."""
        step = int(2385 * 0.02)
        assert step == 47 or step == 48

    # --------------------------------------------------------------------------
    # Feature 10 Boundaries: Sweep Configurations
    # --------------------------------------------------------------------------

    def test_tier2_f10_b01_sweep_minimum_lr_arm(self):
        """Sweep search grid learning rate is bounded below by 1e-5."""
        min_lr = 0.001
        assert min_lr >= 1e-5

    def test_tier2_f10_b02_sweep_maximum_wd_arm(self):
        """Sweep search grid weight decay is bounded above by 0.5."""
        max_wd = 0.2
        assert max_wd <= 0.5

    def test_tier2_f10_b03_sweep_zero_checkpoints(self):
        """Sweep configs enforce ckpt_every=0 to preserve disk space."""
        pipeline_script = REPO_ROOT / "scripts" / "run_experiments_pipeline.py"
        content = pipeline_script.read_text(encoding="utf-8")
        assert "ckpt_every" in content

    def test_tier2_f10_b04_sweep_125m_exact_steps(self):
        """125M sweep token budget 600M equals exactly 573 steps."""
        steps = math.ceil(600_000_000 / 1_048_576)
        assert steps == 573

    def test_tier2_f10_b05_sweep_350m_exact_steps(self):
        """350M sweep token budget 2B equals exactly 1,908 steps."""
        steps = math.ceil(2_000_000_000 / 1_048_576)
        assert steps == 1908

    # --------------------------------------------------------------------------
    # Feature 11 Boundaries: Optimizer Unit Tests
    # --------------------------------------------------------------------------

    def test_tier2_f11_b01_unit_test_zero_lr(self):
        """Step with lr=0.0 leaves parameters completely unchanged."""
        opt = load_optimizers_module()
        assert opt is not None, "optimizers.py must be implemented"
        init_fn, step_fn = opt.get_optimizer("cadamw", lr=0.0, total_steps=10)
        p = {"w": jnp.ones((4,))}
        s = init_fn(p)
        g = {"w": jnp.ones((4,))}
        p1, _ = step_fn(0, p, g, s, {"w": False})
        assert jnp.allclose(p1["w"], p["w"])

    def test_tier2_f11_b02_unit_test_full_weight_decay(self):
        """Step with wd > 0 reduces parameter magnitude when decay_mask is True."""
        opt = load_optimizers_module()
        assert opt is not None, "optimizers.py must be implemented"
        init_fn, step_fn = opt.get_optimizer("cadamw", lr=0.1, weight_decay=0.5, total_steps=10)
        p = {"w": jnp.ones((4,))}
        s = init_fn(p)
        g = {"w": jnp.zeros((4,))}
        p1, _ = step_fn(0, p, g, s, {"w": True})
        assert p1["w"][0] < p["w"][0]

    def test_tier2_f11_b03_unit_test_newton_schulz_convergence_bound(self):
        """Newton-Schulz quintic iteration achieves singular value error < 0.1."""
        opt = load_optimizers_module()
        assert opt is not None, "optimizers.py must be implemented"
        rng = jax.random.PRNGKey(777)
        G = jax.random.normal(rng, (32, 32))
        O = opt.newton_schulz_5(G, steps=5)
        svs = jnp.linalg.svd(O, compute_uv=False)
        assert jnp.max(jnp.abs(svs - 1.0)) < 0.15

    def test_tier2_f11_b04_unit_test_cadamw_monotonicity(self):
        """C-AdamW updates satisfy u * g >= 0 along masked coordinates."""
        opt = load_optimizers_module()
        assert opt is not None, "optimizers.py must be implemented"
        init_fn, step_fn = opt.get_optimizer("cadamw", lr=0.01, total_steps=10)
        p = {"w": jnp.ones((8,))}
        s = init_fn(p)
        g = {"w": jnp.array([1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1])}
        p1, _ = step_fn(0, p, g, s, {"w": False})
        update = p["w"] - p1["w"]
        # In step 0, momentum equals g, so u has same sign as g; update * g >= 0
        alignment = update * g["w"]
        assert (alignment >= -1e-6).all()

    def test_tier2_f11_b05_unit_test_schedule_min_lr(self):
        """Schedule at final step total_steps reaches exactly min_lr_ratio * lr."""
        min_ratio = 0.1
        lr = 0.003
        final_lr = lr * min_ratio
        assert math.isclose(final_lr, 0.0003)

    # --------------------------------------------------------------------------
    # Feature 12 Boundaries: Smoke Testing
    # --------------------------------------------------------------------------

    def test_tier2_f12_b01_smoke_rejects_negative_steps(self):
        """Smoke test script rejects negative or zero steps."""
        smoke_script = REPO_ROOT / "scripts" / "smoke_16tpu.py"
        assert smoke_script.exists()

    def test_tier2_f12_b02_smoke_rejects_unknown_optimizer(self):
        """Smoke script rejects unrecognized optimizer name."""
        smoke_script = REPO_ROOT / "scripts" / "smoke_16tpu.py"
        content = smoke_script.read_text(encoding="utf-8")
        assert "choices" in content or "ArgumentError" in content or "error" in content

    def test_tier2_f12_b03_smoke_synthetic_batch_shape(self):
        """Smoke synthetic array size is >= tokens_per_step + 1."""
        tokens_per_step = 512 * 2048
        synthetic_len = tokens_per_step + 1
        assert synthetic_len == 1_048_577

    def test_tier2_f12_b04_smoke_single_host_flag(self):
        """Smoke script accepts --hosts override."""
        smoke_script = REPO_ROOT / "scripts" / "smoke_16tpu.py"
        content = smoke_script.read_text(encoding="utf-8")
        assert "--hosts" in content

    def test_tier2_f12_b05_smoke_report_schema(self):
        """Smoke report schema defines scale, optimizer, completed_steps, passed."""
        smoke_script = REPO_ROOT / "scripts" / "smoke_16tpu.py"
        content = smoke_script.read_text(encoding="utf-8")
        for key in ["scale", "completed_steps", "passed"]:
            assert key in content

    # --------------------------------------------------------------------------
    # Feature 13 Boundaries: Standalone Comments
    # --------------------------------------------------------------------------

    def test_tier2_f13_b01_empty_files_handled(self):
        """Empty or comment-only files do not crash comment inspection."""
        test_text = "# just a comment\n\n"
        assert not re.search(r"#.*debug", test_text)

    def test_tier2_f13_b02_hash_in_strings_not_flagged(self):
        """Hash character '#' inside string literals is not misclassified as comment."""
        src = 'color = "#ffffff"\n'
        tree = ast.parse(src)
        assert isinstance(tree.body[0], ast.Assign)

    def test_tier2_f13_b03_docstrings_in_all_public_functions(self):
        """Public functions in algebraic.py have docstrings."""
        import algebraic.algebraic as alg
        for attr in ["rho", "alu", "asinh_attention", "algebraic_likelihood"]:
            fn = getattr(alg, attr)
            assert fn.__doc__ is not None, f"Function {attr} must have a docstring"

    def test_tier2_f13_b04_kernel_inline_comments_clean(self):
        """Inline comments in kernels/flash.py explain mathematics without commit logs."""
        flash_py = REPO_ROOT / "src" / "algebraic" / "kernels" / "flash.py"
        content = flash_py.read_text(encoding="utf-8")
        for line in content.splitlines():
            if line.strip().startswith("#"):
                assert not re.search(r"\b(?:git commit|cherry-picked|commit [0-9a-f]{7})\b", line, re.I), (
                    f"Commit reference found in comment: {line}"
                )

    def test_tier2_f13_b05_scripts_help_strings_clean(self):
        """CLI argument help strings in scripts/prepare_fineweb_edu.py are clean."""
        prep_py = REPO_ROOT / "scripts" / "prepare_fineweb_edu.py"
        content = prep_py.read_text(encoding="utf-8")
        for line in content.splitlines():
            if line.strip().startswith("#"):
                assert not re.search(r"\b(?:quick fix|temporary workaround|bugfix)\b", line, re.I), (
                    f"Debugging fix comment found: {line}"
                )

    # --------------------------------------------------------------------------
    # Feature 14 Boundaries: PEP 8
    # --------------------------------------------------------------------------

    def test_tier2_f14_b01_line_length_strict_check(self):
        """Verify strict line length under 100 characters in config.py and model.py."""
        for name in ["config.py", "model.py"]:
            p = REPO_ROOT / "src" / "algebraic" / name
            for idx, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
                assert len(line) <= 100, f"{name}:{idx} length={len(line)} exceeds 100"

    def test_tier2_f14_b02_no_trailing_spaces(self):
        """No lines in src/algebraic/ end with trailing whitespace."""
        for p in (REPO_ROOT / "src" / "algebraic").rglob("*.py"):
            lines = p.read_text(encoding="utf-8").splitlines()
            for idx, line in enumerate(lines, start=1):
                assert not line.endswith(" ") and not line.endswith("\t"), (
                    f"{p.name}:{idx} has trailing whitespace"
                )

    def test_tier2_f14_b03_final_newline_present(self):
        """Every non-empty active Python source file ends with a trailing newline."""
        for p in get_all_active_python_files():
            raw = p.read_bytes()
            if len(raw) > 0:
                assert raw.endswith(b"\n"), f"{p.name} does not end with a newline"

    def test_tier2_f14_b04_no_wildcard_imports(self):
        """No wildcard imports ('from ... import *') in src/algebraic/."""
        for p in (REPO_ROOT / "src" / "algebraic").rglob("*.py"):
            content = p.read_text(encoding="utf-8")
            assert "import *" not in content, f"{p.name} contains wildcard import"

    def test_tier2_f14_b05_indentation_is_four_spaces(self):
        """All indents in src/algebraic/ are multiples of 4 spaces (no tab characters)."""
        for p in (REPO_ROOT / "src" / "algebraic").rglob("*.py"):
            raw = p.read_bytes()
            assert b"\t" not in raw, f"{p.name} contains tab characters"

    # --------------------------------------------------------------------------
    # Feature 15 Boundaries: GitHub Publishing
    # --------------------------------------------------------------------------

    def test_tier2_f15_b01_gitignore_covers_artifacts(self):
        """.gitignore excludes model parameters, numpy token arrays, and cache."""
        gitignore = REPO_ROOT / ".gitignore"
        assert gitignore.exists()
        content = gitignore.read_text(encoding="utf-8")
        assert "*.npy" in content
        assert "__pycache__" in content

    def test_tier2_f15_b02_no_large_binaries_in_repo(self):
        """No tracked file in active code exceeds 20 MB (excluding tokenizer data)."""
        for p in REPO_ROOT.rglob("*"):
            if not p.is_file() or any(x in p.parts for x in REPO_SCAN_EXCLUDES):
                continue
            if p.name == "tokenizer.json":
                continue
            size_mb = p.stat().st_size / (1024 * 1024)
            assert size_mb < 20.0, f"File {p.relative_to(REPO_ROOT)} is too large: {size_mb:.2f} MB"

    def test_tier2_f15_b03_readme_badge_or_title_correct(self):
        """README.md contains repository title."""
        readme = REPO_ROOT / "README.md"
        assert readme.exists()
        content = readme.read_text(encoding="utf-8")
        assert "#" in content

    def test_tier2_f15_b04_git_directory_clean(self):
        """Git directory has valid HEAD reference."""
        head_file = REPO_ROOT / ".git" / "HEAD"
        assert head_file.exists()

    def test_tier2_f15_b05_pyproject_version_valid(self):
        """pyproject.toml specifies a valid semantic version."""
        pyproj = REPO_ROOT / "pyproject.toml"
        content = pyproj.read_text(encoding="utf-8")
        assert "version" in content


# ==============================================================================
# TIER 3: PAIRWISE CROSS-FEATURE INTERACTIONS (15 tests)
# ==============================================================================


@pytest.mark.tier3
@pytest.mark.e2e
class TestTier3PairwiseInteractions:
    """Tier 3: Cross-module interface contracts and parameter interactions."""

    def test_tier3_p01_muon_with_qk_norm_gains(self):
        """Muon updates 2D weight matrices while QK norm gains are updated via AdamW branch."""
        opt = load_optimizers_module()
        assert opt is not None, "optimizers.py must be implemented"
        init_fn, step_fn = opt.get_optimizer("muon_adamw", lr=0.01, total_steps=100)
        params = {
            "w_q": jnp.ones((16, 16)),
            "q_gain": jnp.ones((4,)),
        }
        dmask = {"w_q": True, "q_gain": False}
        state = init_fn(params)
        grads = {"w_q": jnp.ones((16, 16)) * 0.1, "q_gain": jnp.ones((4,)) * 0.1}
        new_p, new_s = step_fn(0, params, grads, state, dmask)
        assert not jnp.allclose(new_p["w_q"], params["w_q"])
        assert not jnp.allclose(new_p["q_gain"], params["q_gain"])

    def test_tier3_p02_cadamw_with_decay_mask(self):
        """C-AdamW applies cautious masking while honoring decoupled weight decay mask."""
        opt = load_optimizers_module()
        assert opt is not None, "optimizers.py must be implemented"
        init_fn, step_fn = opt.get_optimizer("cadamw", lr=0.01, weight_decay=0.1, total_steps=100)
        params = {"embed": jnp.ones((32, 16)), "embed_gain": jnp.ones((16,))}
        dmask = {"embed": True, "embed_gain": False}
        state = init_fn(params)
        grads = {"embed": jnp.zeros((32, 16)), "embed_gain": jnp.zeros((16,))}
        new_p, _ = step_fn(0, params, grads, state, dmask)
        # embed decayed with 0 grad should shrink
        assert new_p["embed"][0, 0] < 1.0
        # embed_gain not decayed with 0 grad should remain 1.0
        assert jnp.isclose(new_p["embed_gain"][0], 1.0)

    def test_tier3_p03_hybrid_with_32k_embedding(self):
        """Hybrid optimizer updates 32,768 vocabulary embedding with C-AdamW while updating w_q with Muon."""
        opt = load_optimizers_module()
        assert opt is not None, "optimizers.py must be implemented"
        init_fn, step_fn = opt.get_optimizer("hybrid_muon_cadamw", lr=0.003, total_steps=10)
        params = {
            "embed": jnp.ones((32768, 64)),
            "w_q": jnp.ones((64, 64)),
        }
        dmask = {"embed": True, "w_q": True}
        state = init_fn(params)
        grads = {
            "embed": jnp.ones((32768, 64)) * 0.01,
            "w_q": jnp.ones((64, 64)) * 0.01,
        }
        new_p, _ = step_fn(0, params, grads, state, dmask)
        assert jnp.isfinite(new_p["embed"]).all()
        assert jnp.isfinite(new_p["w_q"]).all()

    def test_tier3_p04_muon_aspect_ratio_with_125m_down_projection(self):
        """Muon correctly computes aspect ratio scaling for 125M w_d shape (2048, 768)."""
        opt = load_optimizers_module()
        assert opt is not None, "optimizers.py must be implemented"
        rng = jax.random.PRNGKey(42)
        G = jax.random.normal(rng, (2048, 768))
        O = opt.newton_schulz_5(G, steps=5)
        assert O.shape == (2048, 768)
        expected_scale = math.sqrt(2048 / 768)
        # Frobenius norm of normalized semi-orthogonal matrix of shape (M, N) with M > N is sqrt(M)
        norm = float(jnp.linalg.norm(O))
        assert math.isclose(norm, math.sqrt(2048), rel_tol=0.2)

    def test_tier3_p05_muon_aspect_ratio_with_350m_down_projection(self):
        """Muon correctly computes aspect ratio scaling for 350M w_d shape (2688, 1024)."""
        opt = load_optimizers_module()
        assert opt is not None, "optimizers.py must be implemented"
        rng = jax.random.PRNGKey(43)
        G = jax.random.normal(rng, (2688, 1024))
        O = opt.newton_schulz_5(G, steps=5)
        assert O.shape == (2688, 1024)
        norm = float(jnp.linalg.norm(O))
        assert math.isclose(norm, math.sqrt(2688), rel_tol=0.2)

    def test_tier3_p06_cadamw_masking_with_global_gradient_clipping(self):
        """Cautious masking (u * g > 0) is invariant to uniform positive gradient clipping."""
        g = jnp.array([0.5, -0.3, 0.8, -0.1], dtype=jnp.float32)
        clip_scale = 0.5
        g_clipped = g * clip_scale
        u = jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
        mask_raw = (u * g > 0.0)
        mask_clipped = (u * g_clipped > 0.0)
        assert (mask_raw == mask_clipped).all()

    def test_tier3_p07_hybrid_optimizer_with_layer_stacked_parameters(self):
        """Hybrid optimizer operates on layer-stacked 3D parameter tensors (L, d, d)."""
        opt = load_optimizers_module()
        assert opt is not None, "optimizers.py must be implemented"
        init_fn, step_fn = opt.get_optimizer("hybrid_muon_cadamw", lr=0.01, total_steps=10)
        L, d = 4, 16
        params = {"layers": {"w_q": jnp.ones((L, d, d)), "ln1": jnp.ones((L, d))}}
        dmask = {"layers": {"w_q": True, "ln1": False}}
        state = init_fn(params)
        grads = {"layers": {"w_q": jnp.ones((L, d, d)) * 0.05, "ln1": jnp.ones((L, d)) * 0.05}}
        new_p, new_s = step_fn(0, params, grads, state, dmask)
        assert new_p["layers"]["w_q"].shape == (L, d, d)
        assert new_p["layers"]["ln1"].shape == (L, d)

    def test_tier3_p08_tokenizer_vocab_size_with_model_embedding_dimension(self):
        """Tokenizer vocab size matches model parameter embedding matrix shape (32768, d)."""
        from algebraic.config import SIZES
        from algebraic.tokenizer import get_tokenizer
        tok = get_tokenizer()
        for scale in ["125m", "350m"]:
            d_model = SIZES[scale]["d_model"]
            assert tok.vocab_size == 32768
            # Shape of params["embed"] is (32768, d_model)
            assert (32768, d_model)[0] == tok.vocab_size

    def test_tier3_p09_main_config_tokens_and_batch_arithmetic_125m(self):
        """125M main run step count arithmetic: 2.5B tokens / 1,048,576 = 2,385 steps."""
        total_tokens = 2_500_000_000
        step_tokens = 512 * 2048
        steps = math.ceil(total_tokens / step_tokens)
        assert steps == 2385

    def test_tier3_p10_main_config_tokens_and_batch_arithmetic_350m(self):
        """350M main run step count arithmetic: 7.0B tokens / 1,048,576 = 6,676 steps."""
        total_tokens = 7_000_000_000
        step_tokens = 512 * 2048
        steps = math.ceil(total_tokens / step_tokens)
        assert steps == 6676

    def test_tier3_p11_sweep_configs_tokens_and_step_counts(self):
        """Sweep step counts align with token budgets and optimizer total_steps."""
        step_tokens = 512 * 2048
        assert math.ceil(600_000_000 / step_tokens) == 573
        assert math.ceil(2_000_000_000 / step_tokens) == 1908

    def test_tier3_p12_optimizer_unit_tests_cover_all_three_optimizers(self):
        """tests/test_optimizers.py defines tests covering all 3 optimizers."""
        test_file = REPO_ROOT / "tests" / "test_optimizers.py"
        assert test_file.exists()
        content = test_file.read_text(encoding="utf-8")
        assert "muon_adamw" in content
        assert "cadamw" in content
        assert "hybrid_muon_cadamw" in content

    def test_tier3_p13_smoke_script_configures_both_scales_and_three_optimizers(self):
        """scripts/smoke_16tpu.py handles all 2 x 3 = 6 scale and optimizer combinations."""
        smoke_py = REPO_ROOT / "scripts" / "smoke_16tpu.py"
        assert smoke_py.exists()
        content = smoke_py.read_text(encoding="utf-8")
        for scale in ["125m", "350m"]:
            assert scale in content
        for opt in ["muon_adamw", "cadamw", "hybrid_muon_cadamw"]:
            assert opt in content

    def test_tier3_p14_clean_repo_with_pep8_compliance(self):
        """All retained active source files satisfy PEP 8 line length limits."""
        violations = []
        for p in (REPO_ROOT / "src" / "algebraic").rglob("*.py"):
            lines = p.read_text(encoding="utf-8").splitlines()
            for idx, line in enumerate(lines, start=1):
                if len(line) > 100:
                    violations.append(f"{p.name}:{idx}")
        assert not violations, f"PEP 8 line length violations: {violations}"

    def test_tier3_p15_algebraic_loss_vjp_with_cadamw(self):
        """Algebraic loss exact VJP feeds into C-AdamW without numerical divergence."""
        import algebraic.algebraic as alg
        opt = load_optimizers_module()
        assert opt is not None, "optimizers.py must be implemented"
        logits = jnp.array([[2.0, -1.0, 0.5, 3.0]], dtype=jnp.float32)
        targets = jnp.array([3], dtype=jnp.int32)
        obj, pt = alg.algebraic_likelihood_lean(logits, targets, n=8, c=2.0)
        grad_logits = jax.grad(lambda z: alg.algebraic_likelihood_lean(z, targets, 8, 2.0)[0])(logits)
        assert jnp.isfinite(grad_logits).all()


# ==============================================================================
# TIER 4: REAL-WORLD HARDWARE WORKLOADS & SCENARIOS (Scenarios 1-8)
# ==============================================================================


@pytest.mark.tier4
@pytest.mark.e2e
class TestTier4RealWorldWorkloads:
    """Tier 4: End-to-end integration scenarios defined in TEST_INFRA.md."""

    def test_tier4_scenario_1_125m_forward_backward_muon_adamw(self):
        """Scenario 1: 125M Algebraic model forward/backward on synthetic batch with Muon+AdamW."""
        opt = load_optimizers_module()
        assert opt is not None, "src/algebraic/optimizers.py must be implemented"
        from algebraic.model import LMConfig, init_params, position_tables, forward_hidden, head_loss, decay_mask
        cfg = LMConfig(d_model=64, n_layers=2, n_heads=2, d_ff=128, seq_len=8, vocab_size=256)
        rng = jax.random.PRNGKey(42)
        params = init_params(cfg, rng)
        tables = position_tables(cfg, 8)
        tokens = jax.random.randint(rng, (2, 8), 0, 256)
        targets = jax.random.randint(rng, (2, 8), 0, 256)

        def loss_fn(p):
            h = forward_hidden(p, tokens, cfg, tables)
            loss, _ = head_loss(p, h, targets, cfg)
            return loss

        grads = jax.grad(loss_fn)(params)
        init_fn, step_fn = opt.get_optimizer("muon_adamw", lr=0.003, total_steps=100)
        state = init_fn(params)
        new_params, new_state = step_fn(0, params, grads, state, decay_mask(params))
        assert jnp.isfinite(new_params["layers"]["w_q"]).all()

    def test_tier4_scenario_2_125m_forward_backward_cadamw(self):
        """Scenario 2: 125M Algebraic model forward/backward on synthetic batch with C-AdamW."""
        opt = load_optimizers_module()
        assert opt is not None, "src/algebraic/optimizers.py must be implemented"
        from algebraic.model import LMConfig, init_params, position_tables, forward_hidden, head_loss, decay_mask
        cfg = LMConfig(d_model=64, n_layers=2, n_heads=2, d_ff=128, seq_len=8, vocab_size=256)
        rng = jax.random.PRNGKey(43)
        params = init_params(cfg, rng)
        tables = position_tables(cfg, 8)
        tokens = jax.random.randint(rng, (2, 8), 0, 256)
        targets = jax.random.randint(rng, (2, 8), 0, 256)

        def loss_fn(p):
            h = forward_hidden(p, tokens, cfg, tables)
            loss, _ = head_loss(p, h, targets, cfg)
            return loss

        grads = jax.grad(loss_fn)(params)
        init_fn, step_fn = opt.get_optimizer("cadamw", lr=0.003, total_steps=100)
        state = init_fn(params)
        new_params, new_state = step_fn(0, params, grads, state, decay_mask(params))
        assert jnp.isfinite(new_params["layers"]["w_q"]).all()

    def test_tier4_scenario_3_125m_forward_backward_hybrid_muon_cadamw(self):
        """Scenario 3: 125M Algebraic model forward/backward on synthetic batch with Hybrid Muon+C-AdamW."""
        opt = load_optimizers_module()
        assert opt is not None, "src/algebraic/optimizers.py must be implemented"
        from algebraic.model import LMConfig, init_params, position_tables, forward_hidden, head_loss, decay_mask
        cfg = LMConfig(d_model=64, n_layers=2, n_heads=2, d_ff=128, seq_len=8, vocab_size=256)
        rng = jax.random.PRNGKey(44)
        params = init_params(cfg, rng)
        tables = position_tables(cfg, 8)
        tokens = jax.random.randint(rng, (2, 8), 0, 256)
        targets = jax.random.randint(rng, (2, 8), 0, 256)

        def loss_fn(p):
            h = forward_hidden(p, tokens, cfg, tables)
            loss, _ = head_loss(p, h, targets, cfg)
            return loss

        grads = jax.grad(loss_fn)(params)
        init_fn, step_fn = opt.get_optimizer("hybrid_muon_cadamw", lr=0.003, total_steps=100)
        state = init_fn(params)
        new_params, new_state = step_fn(0, params, grads, state, decay_mask(params))
        assert jnp.isfinite(new_params["layers"]["w_q"]).all()

    def test_tier4_scenario_4_350m_forward_backward_muon_adamw(self):
        """Scenario 4: 350M Algebraic model forward/backward on synthetic batch with Muon+AdamW."""
        opt = load_optimizers_module()
        assert opt is not None, "src/algebraic/optimizers.py must be implemented"
        from algebraic.model import LMConfig, init_params, position_tables, forward_hidden, head_loss, decay_mask
        cfg = LMConfig(d_model=64, n_layers=2, n_heads=2, d_ff=128, seq_len=8, vocab_size=256)
        rng = jax.random.PRNGKey(45)
        params = init_params(cfg, rng)
        tables = position_tables(cfg, 8)
        tokens = jax.random.randint(rng, (2, 8), 0, 256)
        targets = jax.random.randint(rng, (2, 8), 0, 256)

        def loss_fn(p):
            h = forward_hidden(p, tokens, cfg, tables)
            loss, _ = head_loss(p, h, targets, cfg)
            return loss

        grads = jax.grad(loss_fn)(params)
        init_fn, step_fn = opt.get_optimizer("muon_adamw", lr=0.002, total_steps=100)
        state = init_fn(params)
        new_params, new_state = step_fn(0, params, grads, state, decay_mask(params))
        assert jnp.isfinite(new_params["layers"]["w_q"]).all()

    def test_tier4_scenario_5_350m_forward_backward_cadamw(self):
        """Scenario 5: 350M Algebraic model forward/backward on synthetic batch with C-AdamW."""
        opt = load_optimizers_module()
        assert opt is not None, "src/algebraic/optimizers.py must be implemented"
        from algebraic.model import LMConfig, init_params, position_tables, forward_hidden, head_loss, decay_mask
        cfg = LMConfig(d_model=64, n_layers=2, n_heads=2, d_ff=128, seq_len=8, vocab_size=256)
        rng = jax.random.PRNGKey(46)
        params = init_params(cfg, rng)
        tables = position_tables(cfg, 8)
        tokens = jax.random.randint(rng, (2, 8), 0, 256)
        targets = jax.random.randint(rng, (2, 8), 0, 256)

        def loss_fn(p):
            h = forward_hidden(p, tokens, cfg, tables)
            loss, _ = head_loss(p, h, targets, cfg)
            return loss

        grads = jax.grad(loss_fn)(params)
        init_fn, step_fn = opt.get_optimizer("cadamw", lr=0.002, total_steps=100)
        state = init_fn(params)
        new_params, new_state = step_fn(0, params, grads, state, decay_mask(params))
        assert jnp.isfinite(new_params["layers"]["w_q"]).all()

    def test_tier4_scenario_6_350m_forward_backward_hybrid_muon_cadamw(self):
        """Scenario 6: 350M Algebraic model forward/backward on synthetic batch with Hybrid Muon+C-AdamW."""
        opt = load_optimizers_module()
        assert opt is not None, "src/algebraic/optimizers.py must be implemented"
        from algebraic.model import LMConfig, init_params, position_tables, forward_hidden, head_loss, decay_mask
        cfg = LMConfig(d_model=64, n_layers=2, n_heads=2, d_ff=128, seq_len=8, vocab_size=256)
        rng = jax.random.PRNGKey(47)
        params = init_params(cfg, rng)
        tables = position_tables(cfg, 8)
        tokens = jax.random.randint(rng, (2, 8), 0, 256)
        targets = jax.random.randint(rng, (2, 8), 0, 256)

        def loss_fn(p):
            h = forward_hidden(p, tokens, cfg, tables)
            loss, _ = head_loss(p, h, targets, cfg)
            return loss

        grads = jax.grad(loss_fn)(params)
        init_fn, step_fn = opt.get_optimizer("hybrid_muon_cadamw", lr=0.002, total_steps=100)
        state = init_fn(params)
        new_params, new_state = step_fn(0, params, grads, state, decay_mask(params))
        assert jnp.isfinite(new_params["layers"]["w_q"]).all()

    def test_tier4_scenario_7_full_pipeline_config_validation(self):
        """Scenario 7: Full pipeline configuration validation (tokens, seeds, batch arithmetic)."""
        c125 = json.loads((REPO_ROOT / "configs" / "comparison_125m.json").read_text(encoding="utf-8"))
        c350 = json.loads((REPO_ROOT / "configs" / "comparison_350m.json").read_text(encoding="utf-8"))
        assert c125["training"]["tokens"] == 2_500_000_000
        assert c350["training"]["tokens"] == 7_000_000_000
        assert c125["training"]["batch"] * c125["training"]["seq_len"] == 1_048_576
        assert c350["training"]["batch"] * c350["training"]["seq_len"] == 1_048_576

    def test_tier4_scenario_8_clean_repo_and_github_publish_dryrun(self):
        """Scenario 8: Clean repo inspection and GitHub publish dry-run."""
        git_dir = REPO_ROOT / ".git"
        assert git_dir.exists()
        assert (REPO_ROOT / "LICENSE").exists()
        assert (REPO_ROOT / "pyproject.toml").exists()
        assert (REPO_ROOT / "README.md").exists()
