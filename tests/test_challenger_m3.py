"""Adversarial stress-test suite for Milestone 3 configurations and pipeline scripts.

Challenger 1 (challenger_m3_1):
1. Offline Safety & Non-Execution Gating.
2. Configuration Invariants & Boundary Validation.
3. Multi-Seed and Seed List Boundaries.
4. Checkpoint Flags and Storage Safety.
5. CLI Robustness & Parameter Combinations.
6. Code Quality & PEP 8 line length compliance.
"""

import json
import math
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.run_comparison import (
    SCALES,
    build_plan,
    parameter_count,
    training_jobs,
)
from scripts.run_experiments_pipeline import (
    get_optimizer_sweep_grid,
    make_main_jobs,
    make_sweep_jobs,
    validate_jobs_recipes,
)


# =============================================================================
# Area 1: Offline Safety & Non-Execution Gating
# =============================================================================

class TestOfflineSafetyAndNonExecution:
    """Empirically prove that default or dry-run execution never triggers cluster
    or network execution.
    """

    def test_run_comparison_dry_run_zero_subprocess(self, monkeypatch):
        """scripts/run_comparison.py without --run never calls subprocess or executes jobs."""
        called = []

        def fake_run(*args, **kwargs):
            called.append(args)
            raise AssertionError(f"subprocess.run invoked unexpectedly: {args}")

        def fake_popen(*args, **kwargs):
            called.append(args)
            raise AssertionError(f"subprocess.Popen invoked unexpectedly: {args}")

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr(subprocess, "Popen", fake_popen)

        from scripts.run_comparison import main

        for scale in ["125m", "350m"]:
            monkeypatch.setattr(sys, "argv", ["run_comparison.py", "--scale", scale])
            ret = main()
            assert ret == 0, f"Expected clean exit 0 for scale {scale}"

        assert len(called) == 0, f"Subprocess was called during dry run: {called}"

    def test_run_comparison_plan_training_launched_flag_is_false(self):
        """Plan dictionary returned by build_plan explicitly sets training_launched=False."""
        plan_125m = build_plan(
            REPO_ROOT / "configs/comparison_125m.json",
            Path("/tmp/dummy_data"),
            Path("/tmp/dummy_evals"),
            Path("/tmp/dummy_recall"),
            "test_group",
            1,
        )
        assert plan_125m["training_launched"] is False

        plan_350m = build_plan(
            REPO_ROOT / "configs/comparison_350m.json",
            Path("/tmp/dummy_data"),
            Path("/tmp/dummy_evals"),
            Path("/tmp/dummy_recall"),
            "test_group",
            1,
        )
        assert plan_350m["training_launched"] is False

    def test_run_experiments_pipeline_dry_run_zero_ssh_or_cluster(self, monkeypatch):
        """Pipeline script dry run never triggers SSH, df, clean_storage, or run_slice."""
        called_cmds = []

        def fake_run(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("cmd")
            called_cmds.append(cmd)
            raise AssertionError(f"Cluster subprocess.run called during dry run: {cmd}")

        def fake_popen(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("cmd")
            called_cmds.append(cmd)
            raise AssertionError(f"Cluster subprocess.Popen called during dry run: {cmd}")

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        monkeypatch.setattr(subprocess, "check_output", fake_run)

        from scripts.run_experiments_pipeline import main

        # Test phase sweep, main, and all under default (dry_run=True)
        for phase in ["sweep", "main", "all"]:
            monkeypatch.setattr(sys, "argv", ["run_experiments_pipeline.py", "--phase", phase])
            main()

        # Test explicit --dry-run
        monkeypatch.setattr(
            sys, "argv", ["run_experiments_pipeline.py", "--phase", "all", "--dry-run"]
        )
        main()

        assert len(called_cmds) == 0, f"Commands executed during dry run: {called_cmds}"

    def test_run_experiments_pipeline_network_safety(self, monkeypatch):
        """Ensure no network connections (sockets) are established during pipeline dry run."""
        import socket

        def fake_connect(*args, **kwargs):
            raise AssertionError(f"Network socket connection attempted: {args}")

        monkeypatch.setattr(socket.socket, "connect", fake_connect)

        from scripts.run_experiments_pipeline import main
        monkeypatch.setattr(
            sys, "argv", ["run_experiments_pipeline.py", "--phase", "all", "--dry-run"]
        )
        main()


# =============================================================================
# Area 2: Configuration Invariants & Boundary Validation
# =============================================================================

class TestConfigurationInvariantsAndBoundaries:
    """Stress-test configuration schemas, token arithmetic, and invariant guards."""

    def test_comparison_exact_token_budget_enforcement(self, tmp_path):
        """build_plan rejects any config where tokens != SCALES[size]."""
        cfg_path = tmp_path / "bad_tokens.json"
        base_cfg = json.loads((REPO_ROOT / "configs/comparison_125m.json").read_text())

        # Off-by-one under
        base_cfg["training"]["tokens"] = 2_500_000_000 - 1
        cfg_path.write_text(json.dumps(base_cfg))
        with pytest.raises(ValueError, match="configured token budget"):
            build_plan(cfg_path, Path("d"), Path("b"), Path("r"), "g", 1)

        # Off-by-one over
        base_cfg["training"]["tokens"] = 2_500_000_000 + 1
        cfg_path.write_text(json.dumps(base_cfg))
        with pytest.raises(ValueError, match="configured token budget"):
            build_plan(cfg_path, Path("d"), Path("b"), Path("r"), "g", 1)

        # Zero tokens
        base_cfg["training"]["tokens"] = 0
        cfg_path.write_text(json.dumps(base_cfg))
        with pytest.raises(ValueError, match="configured token budget"):
            build_plan(cfg_path, Path("d"), Path("b"), Path("r"), "g", 1)

    def test_comparison_invalid_architecture_rejection(self, tmp_path):
        """build_plan strictly rejects non-algebraic architectures."""
        cfg_path = tmp_path / "bad_arch.json"
        base_cfg = json.loads((REPO_ROOT / "configs/comparison_125m.json").read_text())

        for bad_arch in [["standard"], ["algebraic", "standard"], []]:
            base_cfg["architectures"] = bad_arch
            cfg_path.write_text(json.dumps(base_cfg))
            with pytest.raises(ValueError, match="algebraic architecture"):
                build_plan(cfg_path, Path("d"), Path("b"), Path("r"), "g", 1)

    def test_comparison_kernel_implementation_invariants(self, tmp_path):
        """build_plan enforces flash attention and fused head implementations."""
        cfg_path = tmp_path / "bad_kernels.json"
        base_cfg = json.loads((REPO_ROOT / "configs/comparison_125m.json").read_text())

        base_cfg["training"]["attn_impl"] = "vanilla"
        cfg_path.write_text(json.dumps(base_cfg))
        with pytest.raises(ValueError, match="fused kernels"):
            build_plan(cfg_path, Path("d"), Path("b"), Path("r"), "g", 1)

        base_cfg["training"]["attn_impl"] = "flash"
        base_cfg["training"]["head_impl"] = "standard"
        cfg_path.write_text(json.dumps(base_cfg))
        with pytest.raises(ValueError, match="fused kernels"):
            build_plan(cfg_path, Path("d"), Path("b"), Path("r"), "g", 1)

    def test_comparison_unsupported_scales(self, tmp_path):
        """build_plan rejects models not in SCALES ('1b', 'proxy', etc.)."""
        cfg_path = tmp_path / "bad_scale.json"
        base_cfg = json.loads((REPO_ROOT / "configs/comparison_125m.json").read_text())

        for bad_scale in ["1b", "proxy", "7b", ""]:
            base_cfg["training"]["size"] = bad_scale
            cfg_path.write_text(json.dumps(base_cfg))
            with pytest.raises(ValueError, match="configured token budget"):
                build_plan(cfg_path, Path("d"), Path("b"), Path("r"), "g", 1)

    def test_pipeline_sweep_scale_boundary(self):
        """make_sweep_jobs rejects invalid or unsupported scales."""
        with pytest.raises(ValueError, match="Unsupported scale"):
            make_sweep_jobs(scale="1b")
        with pytest.raises(ValueError, match="Unsupported scale"):
            make_sweep_jobs(scale="proxy")
        with pytest.raises(ValueError, match="Unsupported scale"):
            make_sweep_jobs(scale="7b")

    def test_pipeline_grid_optimizer_boundary(self):
        """get_optimizer_sweep_grid rejects unknown optimizers or scales."""
        with pytest.raises(ValueError, match="Unknown scale"):
            get_optimizer_sweep_grid("125m", "adamw")
        with pytest.raises(ValueError, match="Unknown scale"):
            get_optimizer_sweep_grid("125m", "sgd")
        with pytest.raises(ValueError, match="Unknown scale"):
            get_optimizer_sweep_grid("1b", "muon_adamw")

    def test_token_arithmetic_exactness(self):
        """Verify token arithmetic across all standard and boundary token configurations."""
        step_tokens = 512 * 2048  # 1,048,576

        # 125M sweep: 600M tokens
        tokens_125m_sweep = 600_000_000
        steps_125m_sweep = math.ceil(tokens_125m_sweep / step_tokens)
        assert steps_125m_sweep == 573
        assert steps_125m_sweep * step_tokens >= tokens_125m_sweep
        assert (steps_125m_sweep - 1) * step_tokens < tokens_125m_sweep

        # 350M sweep: 2B tokens
        tokens_350m_sweep = 2_000_000_000
        steps_350m_sweep = math.ceil(tokens_350m_sweep / step_tokens)
        assert steps_350m_sweep == 1908
        assert steps_350m_sweep * step_tokens >= tokens_350m_sweep
        assert (steps_350m_sweep - 1) * step_tokens < tokens_350m_sweep

        # 125M main: 2.5B tokens
        tokens_125m_main = 2_500_000_000
        steps_125m_main = math.ceil(tokens_125m_main / step_tokens)
        assert steps_125m_main == 2385
        assert steps_125m_main * step_tokens >= tokens_125m_main
        assert (steps_125m_main - 1) * step_tokens < tokens_125m_main

        # 350M main: 7.0B tokens
        tokens_350m_main = 7_000_000_000
        steps_350m_main = math.ceil(tokens_350m_main / step_tokens)
        assert steps_350m_main == 6676
        assert steps_350m_main * step_tokens >= tokens_350m_main
        assert (steps_350m_main - 1) * step_tokens < tokens_350m_main


# =============================================================================
# Area 3: Multi-Seed & Seed List Boundaries in run_comparison.py
# =============================================================================

class TestMultiSeedAndSeedBoundaries:
    """Stress-test multi-seed handling and boundary seed conditions."""

    def test_multi_seed_configuration_seeds_42_43(self):
        """125M config with seeds [42, 43] generates 2 jobs with distinct data seeds."""
        cfg_125m = json.loads((REPO_ROOT / "configs/comparison_125m.json").read_text())
        assert cfg_125m["training_seeds"] == [42, 43]
        jobs = training_jobs(cfg_125m, Path("data"))
        assert len(jobs) == 2
        assert jobs[0]["name"] == "algebraic_s42"
        assert jobs[1]["name"] == "algebraic_s43"

        # Check argument tokens for seed and data-seed
        args_0 = jobs[0]["args"]
        args_1 = jobs[1]["args"]
        assert args_0[args_0.index("--seed") + 1] == "42"
        assert args_0[args_0.index("--data-seed") + 1] == "42"
        assert args_1[args_1.index("--seed") + 1] == "43"
        assert args_1[args_1.index("--data-seed") + 1] == "43"

    def test_single_seed_configuration_seed_42(self):
        """350M config with seeds [42] generates 1 job with configured data seed."""
        cfg_350m = json.loads((REPO_ROOT / "configs/comparison_350m.json").read_text())
        assert cfg_350m["training_seeds"] == [42]
        jobs = training_jobs(cfg_350m, Path("data"))
        assert len(jobs) == 1
        assert jobs[0]["name"] == "algebraic_s42"
        args_0 = jobs[0]["args"]
        assert args_0[args_0.index("--seed") + 1] == "42"
        assert args_0[args_0.index("--data-seed") + 1] == "42"

    def test_empty_seed_list_rejected(self, tmp_path):
        """Empty seed list [] is rejected by build_plan."""
        cfg_path = tmp_path / "empty_seeds.json"
        base_cfg = json.loads((REPO_ROOT / "configs/comparison_125m.json").read_text())
        base_cfg["training_seeds"] = []
        cfg_path.write_text(json.dumps(base_cfg))

        with pytest.raises(ValueError, match="at least one seed"):
            build_plan(cfg_path, Path("d"), Path("b"), Path("r"), "g", 1)

    def test_arbitrary_multi_seeds(self, tmp_path):
        """Arbitrary 3-seed list [101, 102, 103] creates 3 jobs with corresponding seeds."""
        cfg_path = tmp_path / "three_seeds.json"
        base_cfg = json.loads((REPO_ROOT / "configs/comparison_125m.json").read_text())
        base_cfg["training_seeds"] = [101, 102, 103]
        cfg_path.write_text(json.dumps(base_cfg))

        plan = build_plan(cfg_path, Path("d"), Path("b"), Path("r"), "g", 1)
        assert len(plan["jobs"]) == 3
        job_names = [j["name"] for j in plan["jobs"]]
        assert job_names == ["algebraic_s101", "algebraic_s102", "algebraic_s103"]
        data_seeds = [j["args"][j["args"].index("--data-seed") + 1] for j in plan["jobs"]]
        assert data_seeds == ["101", "102", "103"]


# =============================================================================
# Area 4: Checkpoint Flags and Storage Safety
# =============================================================================

class TestCheckpointFlagsAndStorageSafety:
    """Verify checkpoint suppression on all sweep recipes to protect worker disk quotas."""

    def test_all_sweep_jobs_enforce_ckpt_every_zero(self):
        """Every sweep arm across 125M and 350M must specify --ckpt-every 0."""
        for scale in ["125m", "350m"]:
            jobs = make_sweep_jobs(scale=scale)
            assert len(jobs) == 15, f"Expected 15 sweep arms for {scale}, found {len(jobs)}"
            for job in jobs:
                assert job["ckpt_every"] == 0, f"Job {job['name']} ckpt_every != 0"
                args = job["args"]
                assert "--ckpt-every" in args, f"Job {job['name']} missing --ckpt-every"
                idx = args.index("--ckpt-every")
                assert args[idx + 1] == "0", f"Job {job['name']} ckpt_every arg != 0"
                assert "--save-params" not in args, (
                    f"Job {job['name']} should not save full parameters during sweep"
                )

    def test_train_arg_parser_accepts_ckpt_every_zero(self):
        """src/algebraic/train.py parser cleanly handles --ckpt-every 0 without error."""
        from algebraic.train import get_args
        args = get_args(["--out", "/tmp/dummy", "--ckpt-every", "0"])
        assert args.ckpt_every == 0

    def test_train_arg_parser_rejects_negative_ckpt_every(self):
        """src/algebraic/train.py main() rejects negative ckpt_every values with ValueError."""
        from algebraic.train import main
        with pytest.raises(ValueError, match="checkpoint retention"):
            main(["--out", "/tmp/dummy", "--ckpt-every", "-1"])



# =============================================================================
# Area 5: CLI Robustness & Parameter Combinations
# =============================================================================

class TestCLIRobustnessAndParameterCombinations:
    """Stress-test CLI parsers with valid and invalid options."""

    def test_run_experiments_pipeline_invalid_phase(self):
        """scripts/run_experiments_pipeline.py rejects invalid --phase values with exit code 2."""
        script = str(REPO_ROOT / "scripts/run_experiments_pipeline.py")
        res = subprocess.run(
            [sys.executable, script, "--phase", "invalid"],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 2
        assert "invalid choice" in res.stderr

    def test_run_experiments_pipeline_invalid_scale(self):
        """scripts/run_experiments_pipeline.py rejects invalid --scale values with exit code 2."""
        script = str(REPO_ROOT / "scripts/run_experiments_pipeline.py")
        res = subprocess.run(
            [sys.executable, script, "--scale", "1b"],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 2
        assert "invalid choice" in res.stderr

    def test_run_experiments_pipeline_unrecognized_optimizer_flag(self):
        """scripts/run_experiments_pipeline.py rejects unrecognized --optimizer CLI argument."""
        script = str(REPO_ROOT / "scripts/run_experiments_pipeline.py")
        res = subprocess.run(
            [sys.executable, script, "--optimizer", "cadamw"],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 2
        assert "unrecognized arguments" in res.stderr

    def test_run_comparison_invalid_optimizer(self):
        """scripts/run_comparison.py rejects invalid --optimizer choices."""
        res = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts/run_comparison.py"), "--optimizer", "adam"],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 2
        assert "invalid choice" in res.stderr

    def test_run_comparison_valid_optimizer_flags(self):
        """scripts/run_comparison.py correctly parses and forwards all supported optimizer flags."""
        for opt in ["cadamw", "hybrid_muon_cadamw", "muon_adamw"]:
            res = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/run_comparison.py"),
                    "--scale", "125m",
                    "--optimizer", opt,
                    "--muon-lr", "0.025",
                    "--muon-wd", "0.08",
                    "--cautious-eps", "0.002",
                ],
                capture_output=True,
                text=True,
            )
            assert res.returncode == 0, f"Failed for optimizer {opt}: {res.stderr}"
            plan = json.loads(res.stdout)
            for job in plan["jobs"]:
                args = job["args"]
                assert args[args.index("--optimizer") + 1] == opt
                assert args[args.index("--muon-lr") + 1] == "0.025"
                assert args[args.index("--muon-wd") + 1] == "0.08"
                assert args[args.index("--cautious-eps") + 1] == "0.002"
                if opt in ("muon_adamw", "hybrid_muon_cadamw"):
                    assert "--muon-momentum" in args
                    assert "--muon-steps" in args
                else:
                    assert "--muon-momentum" not in args
                    assert "--muon-steps" not in args

    def test_run_comparison_invalid_group(self, tmp_path):
        """build_plan rejects groups with path separators or invalid characters."""
        cfg_125m = REPO_ROOT / "configs/comparison_125m.json"
        with pytest.raises(ValueError, match="simple directory name"):
            build_plan(cfg_125m, Path("d"), Path("b"), Path("r"), "invalid/group/path", 1)

    def test_run_comparison_invalid_eval_batch(self):
        """build_plan rejects eval_batch <= 0."""
        cfg_125m = REPO_ROOT / "configs/comparison_125m.json"
        with pytest.raises(ValueError, match="evaluation batch must be positive"):
            build_plan(cfg_125m, Path("d"), Path("b"), Path("r"), "valid_group", 0)

    def test_run_comparison_jobs_out_writing(self, tmp_path):
        """scripts/run_comparison.py writes valid jobs to --jobs-out without running."""
        out_jobs = tmp_path / "subdir" / "jobs.json"
        res = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/run_comparison.py"),
                "--scale", "125m",
                "--jobs-out", str(out_jobs),
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        assert out_jobs.exists()
        jobs_data = json.loads(out_jobs.read_text())
        assert len(jobs_data) == 2
        assert jobs_data[0]["name"] == "algebraic_s42"
        assert jobs_data[1]["name"] == "algebraic_s43"

    def test_read_selection_adversarial_inputs(self, tmp_path):
        """read_selection rejects non-finite, negative, incomplete, or mismatched selections."""
        import hashlib
        from scripts.run_comparison import read_selection

        cfg_file = REPO_ROOT / "configs/comparison_125m.json"
        expected_hash = hashlib.sha256(cfg_file.read_bytes()).hexdigest()
        sel_file = tmp_path / "sel.json"

        # Case 1: Hash mismatch
        sel_file.write_text(json.dumps({
            "comparison_config_sha256": "bad_hash",
            "complete": True,
            "selected": {"algebraic": {"lr": 0.003, "wd": 0.1}},
        }))
        with pytest.raises(ValueError, match="does not match"):
            read_selection(sel_file, cfg_file)

        # Case 2: Incomplete
        sel_file.write_text(json.dumps({
            "comparison_config_sha256": expected_hash,
            "complete": False,
            "selected": {"algebraic": {"lr": 0.003, "wd": 0.1}},
        }))
        with pytest.raises(ValueError, match="does not match"):
            read_selection(sel_file, cfg_file)

        # Case 3: Negative lr
        sel_file.write_text(json.dumps({
            "comparison_config_sha256": expected_hash,
            "complete": True,
            "selected": {"algebraic": {"lr": -0.001, "wd": 0.1}},
        }))
        with pytest.raises(ValueError, match="invalid selected learning rate"):
            read_selection(sel_file, cfg_file)

        # Case 4: Non-finite wd (inf)
        sel_file.write_text(json.dumps({
            "comparison_config_sha256": expected_hash,
            "complete": True,
            "selected": {"algebraic": {"lr": 0.003, "wd": float("inf")}},
        }))
        with pytest.raises(ValueError, match="invalid selected learning rate"):
            read_selection(sel_file, cfg_file)


# =============================================================================
# Area 6: Code Quality & Line Length Audit
# =============================================================================

class TestCodeQualityAndLineLength:
    """Verify that all edited files comply strictly with PEP 8 and line length <= 100."""

    def test_line_length_under_100(self):
        files_to_check = [
            REPO_ROOT / "configs/comparison_125m.json",
            REPO_ROOT / "configs/comparison_350m.json",
            REPO_ROOT / "scripts/run_comparison.py",
            REPO_ROOT / "scripts/run_experiments_pipeline.py",
        ]
        violations = []
        for fpath in files_to_check:
            with open(fpath, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    stripped = line.rstrip("\r\n")
                    if len(stripped) > 100:
                        violations.append((
                            str(fpath.relative_to(REPO_ROOT)), line_no, len(stripped), stripped
                        ))
        assert not violations, f"Found lines exceeding 100 characters: {violations}"

