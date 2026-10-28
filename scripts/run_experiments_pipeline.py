#!/usr/bin/env python3
"""Unified pipeline for 125M and 350M hyperparameter sweeps and main training runs.

TPU Pod Slice: Google Cloud TPU v4-32 (16 chips, 4 workers: w0, w1, w2, w3).

Workflow:
1. Phase 1: Hyperparameter sweep on literature-derived grids:
   - 125M Algebraic Transformer: 600M tokens per arm (573 steps), seed 42.
   - 350M Algebraic Transformer: 2B tokens per arm (1,908 steps), seed 42.
   - Three pure XLA optimizers:
     * Classic Muon (2D weights) + AdamW (1D weights, cosine schedule)
     * Pure Cautious AdamW (C-AdamW) with update alignment masking
     * Hybrid Muon (2D weights) + C-AdamW (1D weights)
   - Evaluated on held-out validation window (8M tokens) every 100/250 steps.
   - Zero-checkpoint retention (ckpt_every=0) during sweep to preserve disk space.
2. Phase 2: Main training runs using best hyperparameters:
   - 125M Algebraic Transformer on 2.5B tokens across 2 seeds (42, 43) -> 2,385 steps.
   - 350M Algebraic Transformer on 7.0B tokens across 1 seed (42) -> 6,676 steps.
   - Periodic checkpoint retention and parameter export.
3. Verification and Safety:
   - Configure-only / dry-run mode (--dry-run) validates all job recipes without cluster launch.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List

ROOT = Path(__file__).resolve()
while ROOT.parent != ROOT and not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

DATA_DIR = Path.home() / "data" / "fw25"
DATA_DIR_7B = Path.home() / "data" / "fw7b"
HOSTS = ("w0", "w1", "w2", "w3")
IST = timezone(timedelta(hours=5, minutes=30))
VOCAB_SIZE = 32768


def ist_now_str() -> str:
    """Return formatted current timestamp in IST."""
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def get_disk_info(hosts=HOSTS) -> dict[str, str]:
    """Query available disk storage across cluster hosts."""
    info = {}
    for h in hosts:
        try:
            line = subprocess.check_output(
                ["ssh", h, "df -h / | tail -n 1"], text=True, timeout=15
            ).strip()
            parts = line.split()
            info[h] = f"Size: {parts[1]}, Used: {parts[2]}, Avail: {parts[3]} ({parts[4]})"
        except Exception as e:
            info[h] = f"Error: {e}"
    return info


def clean_storage(hosts=HOSTS, prune_run_dir=None) -> None:
    """Clean transient log files and pruned checkpoint directories on cluster hosts."""
    for h in hosts:
        cmds = [
            "rm -rf /tmp/pytest* /tmp/tpu_logs/* /tmp/*.log 2>/dev/null || true",
        ]
        if prune_run_dir:
            cmds.append(f"rm -rf {shlex.quote(str(prune_run_dir))}/ckpt || true")
        cmd_str = " ; ".join(cmds)
        subprocess.run(["ssh", h, cmd_str], timeout=30, check=False)


def run_command_stream(cmd: list[str], log_file: Path | None = None) -> None:
    """Run subprocess and stream output to console and optional log file."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    log_fh = open(log_file, "a") if log_file else None
    try:
        if proc.stdout:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                if log_fh:
                    log_fh.write(line)
                    log_fh.flush()
    finally:
        proc.wait()
        if log_fh:
            log_fh.close()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {cmd}")


# -----------------------------------------------------------------------------
# Hyperparameter Grid Definitions (Literature Grounded)
# -----------------------------------------------------------------------------

def get_optimizer_sweep_grid(scale: str, optimizer: str) -> list[dict[str, Any]]:
    """Return literature-derived hyperparameter candidate arms for given scale and optimizer.

    Grids follow:
    - Muon: Keller Jordan (2024), Moonshot AI Moonlight (arXiv:2502.16982).
    - C-AdamW: Kaizhao Liang et al., ICLR 2025 (arXiv:2411.16085).
    """
    if scale == "125m":
        if optimizer == "muon_adamw":
            return [
                {"cand_name": "base", "muon_lr": 0.02, "lr": 0.003, "muon_wd": 0.05, "wd": 0.1},
                {"cand_name": "low_lr", "muon_lr": 0.01, "lr": 0.0015, "muon_wd": 0.05, "wd": 0.1},
                {"cand_name": "high_lr", "muon_lr": 0.04, "lr": 0.003, "muon_wd": 0.05, "wd": 0.1},
                {"cand_name": "low_wd", "muon_lr": 0.02, "lr": 0.003, "muon_wd": 0.01, "wd": 0.1},
                {"cand_name": "high_wd", "muon_lr": 0.02, "lr": 0.003, "muon_wd": 0.10, "wd": 0.1},
            ]
        elif optimizer == "cadamw":
            return [
                {"cand_name": "base", "lr": 0.0030, "wd": 0.10, "cautious_eps": 1e-3},
                {"cand_name": "low_lr", "lr": 0.0015, "wd": 0.10, "cautious_eps": 1e-3},
                {"cand_name": "high_lr", "lr": 0.0045, "wd": 0.10, "cautious_eps": 1e-3},
                {"cand_name": "low_wd", "lr": 0.0030, "wd": 0.05, "cautious_eps": 1e-3},
                {"cand_name": "high_wd", "lr": 0.0030, "wd": 0.20, "cautious_eps": 1e-3},
            ]
        elif optimizer == "hybrid_muon_cadamw":
            return [
                {
                    "cand_name": "base",
                    "muon_lr": 0.02,
                    "lr": 0.003,
                    "muon_wd": 0.05,
                    "wd": 0.10,
                    "cautious_eps": 1e-3,
                },
                {
                    "cand_name": "low_lr",
                    "muon_lr": 0.01,
                    "lr": 0.0015,
                    "muon_wd": 0.05,
                    "wd": 0.10,
                    "cautious_eps": 1e-3,
                },
                {
                    "cand_name": "high_lr",
                    "muon_lr": 0.04,
                    "lr": 0.003,
                    "muon_wd": 0.05,
                    "wd": 0.10,
                    "cautious_eps": 1e-3,
                },
                {
                    "cand_name": "low_wd",
                    "muon_lr": 0.02,
                    "lr": 0.003,
                    "muon_wd": 0.01,
                    "wd": 0.10,
                    "cautious_eps": 1e-3,
                },
                {
                    "cand_name": "high_wd",
                    "muon_lr": 0.02,
                    "lr": 0.003,
                    "muon_wd": 0.10,
                    "wd": 0.10,
                    "cautious_eps": 1e-3,
                },
            ]
    elif scale == "350m":
        if optimizer == "muon_adamw":
            return [
                {"cand_name": "base", "muon_lr": 0.015, "lr": 0.002, "muon_wd": 0.05, "wd": 0.1},
                {"cand_name": "low_lr", "muon_lr": 0.008, "lr": 0.001, "muon_wd": 0.05, "wd": 0.1},
                {"cand_name": "high_lr", "muon_lr": 0.030, "lr": 0.002, "muon_wd": 0.05, "wd": 0.1},
                {"cand_name": "low_wd", "muon_lr": 0.015, "lr": 0.002, "muon_wd": 0.01, "wd": 0.1},
                {"cand_name": "high_wd", "muon_lr": 0.015, "lr": 0.002, "muon_wd": 0.10, "wd": 0.1},
            ]
        elif optimizer == "cadamw":
            return [
                {"cand_name": "base", "lr": 0.0020, "wd": 0.10, "cautious_eps": 1e-3},
                {"cand_name": "low_lr", "lr": 0.0010, "wd": 0.10, "cautious_eps": 1e-3},
                {"cand_name": "high_lr", "lr": 0.0030, "wd": 0.10, "cautious_eps": 1e-3},
                {"cand_name": "low_wd", "lr": 0.0020, "wd": 0.05, "cautious_eps": 1e-3},
                {"cand_name": "high_wd", "lr": 0.0020, "wd": 0.20, "cautious_eps": 1e-3},
            ]
        elif optimizer == "hybrid_muon_cadamw":
            return [
                {
                    "cand_name": "base",
                    "muon_lr": 0.015,
                    "lr": 0.002,
                    "muon_wd": 0.05,
                    "wd": 0.10,
                    "cautious_eps": 1e-3,
                },
                {
                    "cand_name": "low_lr",
                    "muon_lr": 0.008,
                    "lr": 0.001,
                    "muon_wd": 0.05,
                    "wd": 0.10,
                    "cautious_eps": 1e-3,
                },
                {
                    "cand_name": "high_lr",
                    "muon_lr": 0.030,
                    "lr": 0.002,
                    "muon_wd": 0.05,
                    "wd": 0.10,
                    "cautious_eps": 1e-3,
                },
                {
                    "cand_name": "low_wd",
                    "muon_lr": 0.015,
                    "lr": 0.002,
                    "muon_wd": 0.01,
                    "wd": 0.10,
                    "cautious_eps": 1e-3,
                },
                {
                    "cand_name": "high_wd",
                    "muon_lr": 0.015,
                    "lr": 0.002,
                    "muon_wd": 0.10,
                    "wd": 0.10,
                    "cautious_eps": 1e-3,
                },
            ]
    raise ValueError(f"Unknown scale '{scale}' or optimizer '{optimizer}'")


# -----------------------------------------------------------------------------
# Sweep Job Generator
# -----------------------------------------------------------------------------

def make_sweep_jobs(
    scale: str = "125m",
    optimizers: list[str] | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build list of hyperparameter sweep jobs.

    - 125M: 600M tokens (600_000_000), 573 steps, batch 512, micro 4, seq_len 2048.
    - 350M: 2B tokens (2_000_000_000), 1,908 steps, batch 512, micro 2, seq_len 2048.
    - Zero checkpointing (ckpt_every=0) to preserve disk quota.
    """
    if optimizers is None:
        optimizers = ["muon_adamw", "cadamw", "hybrid_muon_cadamw"]

    seq_len = 2048
    batch = 512
    step_tokens = batch * seq_len  # 1,048,576

    if scale == "125m":
        tokens_budget = 600_000_000
        micro = 4
        eval_every = 100
        sweep_cand = DATA_DIR / "fineweb_sweep_600M.npy"
        train_file = str(sweep_cand if sweep_cand.is_file() else DATA_DIR / "fineweb_train_2_5B.npy")
    elif scale == "350m":
        tokens_budget = 2_000_000_000
        micro = 2
        eval_every = 250
        train_file = str(DATA_DIR_7B / "fineweb_train_7B.npy")
    else:
        raise ValueError(f"Unsupported scale: {scale}")

    steps = math.ceil(tokens_budget / step_tokens)
    valid_file = str(DATA_DIR / "fineweb_valid.npy")

    jobs = []
    for opt in optimizers:
        grid = get_optimizer_sweep_grid(scale, opt)
        for arm in grid:
            cand_name = arm["cand_name"]
            job_name = f"{scale}_{opt}_{cand_name}_s{seed}"
            lr = arm.get("lr", 0.003)
            wd = arm.get("wd", 0.1)

            args = [
                "--preset", "algebraic",
                "--size", scale,
                "--seq-len", str(seq_len),
                "--batch", str(batch),
                "--micro", str(micro),
                "--tokens", str(tokens_budget),
                "--steps", str(steps),
                "--optimizer", opt,
                "--lr", str(lr),
                "--wd", str(wd),
                "--warmup", "0.02",
                "--min-lr-ratio", "0.1",
                "--b1", "0.9",
                "--b2", "0.95",
                "--adam-eps", "1e-8",
                "--clip", "1.0",
                "--schedule", "cosine",
                "--seed", str(seed),
                "--train", train_file,
                "--valid", valid_file,
                "--eval-tokens", "8000000",
                "--eval-batch", str(micro),
                "--eval-every", str(eval_every),
                "--log-every", "25",
                "--ckpt-every", "0",
                "--keep-ckpts", "1",
                "--set", f"vocab_size={VOCAB_SIZE}",
                "--set", "attn_impl=flash",
                "--set", "head_impl=fused",
            ]

            if "muon_lr" in arm:
                args.extend(["--muon-lr", str(arm["muon_lr"])])
            if "muon_wd" in arm:
                args.extend(["--muon-wd", str(arm["muon_wd"])])
            if opt in ("muon_adamw", "hybrid_muon_cadamw"):
                args.extend(["--muon-momentum", "0.95", "--muon-steps", "5"])
            if "cautious_eps" in arm:
                args.extend(["--cautious-eps", str(arm["cautious_eps"])])

            jobs.append({
                "name": job_name,
                "scale": scale,
                "optimizer": opt,
                "cand_name": cand_name,
                "lr": lr,
                "wd": wd,
                "muon_lr": arm.get("muon_lr"),
                "muon_wd": arm.get("muon_wd"),
                "tokens": tokens_budget,
                "steps": steps,
                "ckpt_every": 0,
                "args": args,
            })

    return jobs


# -----------------------------------------------------------------------------
# Main Training Job Generator (Phase 2)
# -----------------------------------------------------------------------------

def make_main_jobs(
    selected_configs: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build list of main comparison runs.

    - 125M on 2.5B tokens (2_500_000_000, 2,385 steps) across 2 seeds: (42, 43).
    - 350M on 7.0B tokens (7_000_000_000, 6,676 steps) across 1 seed: (42).
    """
    runs_spec = [
        {
            "scale": "125m",
            "tokens": 2_500_000_000,
            "steps": 2385,
            "seeds": (42, 43),
            "micro": 4,
            "train_file": str(DATA_DIR / "fineweb_train_2_5B.npy"),
        },
        {
            "scale": "350m",
            "tokens": 7_000_000_000,
            "steps": 6676,
            "seeds": (42,),
            "micro": 2,
            "train_file": str(DATA_DIR_7B / "fineweb_train_7B.npy"),
        },
    ]

    default_selected = {
        "125m": {
            "optimizer": "hybrid_muon_cadamw",
            "lr": 0.003,
            "wd": 0.1,
            "muon_lr": 0.02,
            "muon_wd": 0.05,
        },
        "350m": {
            "optimizer": "hybrid_muon_cadamw",
            "lr": 0.002,
            "wd": 0.1,
            "muon_lr": 0.015,
            "muon_wd": 0.05,
        },
    }
    configs = selected_configs or default_selected

    jobs = []
    for spec in runs_spec:
        scale = spec["scale"]
        cfg = configs.get(scale, default_selected[scale])
        opt = cfg.get("optimizer", "hybrid_muon_cadamw")
        lr = cfg.get("lr", 0.003)
        wd = cfg.get("wd", 0.1)

        for seed in spec["seeds"]:
            name = f"main_{scale}_{opt}_s{seed}"
            args = [
                "--preset", "algebraic",
                "--size", scale,
                "--seq-len", "2048",
                "--batch", "512",
                "--micro", str(spec["micro"]),
                "--tokens", str(spec["tokens"]),
                "--steps", str(spec["steps"]),
                "--optimizer", opt,
                "--lr", str(lr),
                "--wd", str(wd),
                "--warmup", "0.02",
                "--min-lr-ratio", "0.1",
                "--b1", "0.9",
                "--b2", "0.95",
                "--adam-eps", "1e-8",
                "--clip", "1.0",
                "--schedule", "cosine",
                "--seed", str(seed),
                "--train", spec["train_file"],
                "--valid", str(DATA_DIR / "fineweb_valid.npy"),
                "--eval-tokens", "8000000",
                "--eval-batch", str(spec["micro"]),
                "--eval-every", "250",
                "--ckpt-every", "200",
                "--keep-ckpts", "2",
                "--log-every", "25",
                "--resume",
                "--save-params",
                "--set", f"vocab_size={VOCAB_SIZE}",
                "--set", "attn_impl=flash",
                "--set", "head_impl=fused",
            ]
            if "muon_lr" in cfg:
                args.extend(["--muon-lr", str(cfg["muon_lr"])])
            if "muon_wd" in cfg:
                args.extend(["--muon-wd", str(cfg["muon_wd"])])
            if opt in ("muon_adamw", "hybrid_muon_cadamw"):
                args.extend(["--muon-momentum", "0.95", "--muon-steps", "5"])
            if "cautious_eps" in cfg:
                args.extend(["--cautious-eps", str(cfg["cautious_eps"])])

            jobs.append({
                "name": name,
                "scale": scale,
                "optimizer": opt,
                "seed": seed,
                "tokens": spec["tokens"],
                "steps": spec["steps"],
                "ckpt_every": 200,
                "args": args,
            })
    return jobs


# -----------------------------------------------------------------------------
# Configuration Validation (Offline / Dry-Run)
# -----------------------------------------------------------------------------

def validate_jobs_recipes(jobs: list[dict[str, Any]]) -> None:
    """Validate argument syntax and model dimensions for each job recipe offline."""
    from algebraic.config import build
    from algebraic.train import get_args, parse_overrides

    for j in jobs:
        parsed = get_args([*j["args"], "--out", "/tmp/validation_dummy"])
        overrides = parse_overrides(parsed.set)
        cfg = build(parsed.preset, parsed.size, **overrides)
        assert cfg.vocab_size == VOCAB_SIZE, f"Vocab size mismatch for {j['name']}"
        tokens_per_step = parsed.batch * parsed.seq_len
        calculated_steps = math.ceil(parsed.tokens / tokens_per_step)
        assert calculated_steps == j["steps"], f"Step calculation mismatch for {j['name']}"


# -----------------------------------------------------------------------------
# Phase Runners
# -----------------------------------------------------------------------------

def run_phase1_sweep(scale: str = "all", dry_run: bool = True) -> dict[str, Any]:
    """Execute or dry-run Phase 1 hyperparameter sweeps."""
    print(f"\n=======================================================", flush=True)
    print(f"[{ist_now_str()}] PHASE 1: HYPERPARAMETER SWEEP CONFIGURATION", flush=True)
    print(f"=======================================================\n", flush=True)

    scales = ["125m", "350m"] if scale == "all" else [scale]
    all_jobs: list[dict[str, Any]] = []
    summary_records: dict[str, Any] = {}

    for sc in scales:
        jobs = make_sweep_jobs(scale=sc)
        all_jobs.extend(jobs)
        group = f"hparam_sweep_{sc}"
        job_dir = ROOT / "runs" / group
        job_dir.mkdir(parents=True, exist_ok=True)
        job_file = job_dir / "jobs.json"
        job_data = [{"name": j["name"], "args": j["args"]} for j in jobs]
        job_file.write_text(json.dumps(job_data, indent=2))

        step_count = 573 if sc == "125m" else 1908
        token_str = "600M" if sc == "125m" else "2B"
        print(
            f"Configured {len(jobs)} arms for scale {sc} "
            f"({token_str} tokens/arm, {step_count} steps, ckpt_every=0):"
        )
        for j in jobs:
            print(
                f"  - {j['name']}: opt={j['optimizer']}, "
                f"cand={j['cand_name']}, lr={j['lr']}, wd={j['wd']}"
            )

        summary_records[sc] = {
            "scale": sc,
            "arms": len(jobs),
            "step_count": step_count,
            "jobs": [j["name"] for j in jobs],
        }

    validate_jobs_recipes(all_jobs)
    print(f"\nSuccessfully validated {len(all_jobs)} sweep recipes across scales {scales}.")

    if dry_run:
        print(
            "[DRY-RUN] Sweep job configurations generated and verified. Full execution omitted.",
            flush=True,
        )
        return summary_records

    # Execution path across TPU cluster slice
    clean_storage(HOSTS)
    print("Initial disk info across workers:")
    for h, s in get_disk_info(HOSTS).items():
        print(f"  {h}: {s}")

    for sc in scales:
        group = f"hparam_sweep_{sc}"
        job_path = ROOT / "runs" / group / "jobs.json"
        cmd = [
            sys.executable, str(ROOT / "scripts/run_slice.py"),
            str(job_path),
            "--group", group,
            "--hosts", ",".join(HOSTS),
            "--skip-done",
        ]
        run_command_stream(cmd)

    return summary_records


def run_phase2_main(dry_run: bool = True) -> dict[str, Any]:
    """Execute or dry-run Phase 2 main comparison training runs."""
    print(f"\n=======================================================", flush=True)
    print(f"[{ist_now_str()}] PHASE 2: MAIN TRAINING RUN CONFIGURATION", flush=True)
    print(f"=======================================================\n", flush=True)

    jobs = make_main_jobs()
    job_dir = ROOT / "runs" / "main_runs"
    job_dir.mkdir(parents=True, exist_ok=True)
    job_file = job_dir / "jobs.json"
    job_data = [{"name": j["name"], "args": j["args"]} for j in jobs]
    job_file.write_text(json.dumps(job_data, indent=2))

    print(f"Configured {len(jobs)} main training runs:")
    for j in jobs:
        print(
            f"  - {j['name']}: {j['scale']} on {j['tokens']} tokens "
            f"({j['steps']} steps, seed {j['seed']})"
        )

    validate_jobs_recipes(jobs)
    print(f"\nSuccessfully validated {len(jobs)} main training run recipes.")

    if dry_run:
        print(
            "[DRY-RUN] Main run configurations generated and verified. Full execution omitted.",
            flush=True,
        )
        return {"main_jobs": [j["name"] for j in jobs]}

    # Main run execution path
    for j in jobs:
        name = j["name"]
        single_job_path = ROOT / "runs" / "main_runs" / f"job_{name}.json"
        single_job_path.write_text(json.dumps([{"name": j["name"], "args": j["args"]}], indent=2))
        cmd = [
            sys.executable, str(ROOT / "scripts/run_slice.py"),
            str(single_job_path),
            "--group", "main_runs",
            "--hosts", ",".join(HOSTS),
            "--skip-done",
        ]
        run_command_stream(cmd)

    return {"main_jobs": [j["name"] for j in jobs]}


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase",
        choices=("all", "sweep", "main"),
        default="sweep",
        help="Pipeline phase: sweep, main, or all",
    )
    p.add_argument(
        "--scale",
        choices=("all", "125m", "350m"),
        default="all",
        help="Model scale: 125m, 350m, or all",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Generate and validate configs without launching TPU cluster execution",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Execute cluster jobs across TPU slice (safety flag)",
    )
    args = p.parse_args()

    # Safety: execution requires explicit --execute flag
    is_dry_run = not args.execute

    if args.phase in ("all", "sweep"):
        run_phase1_sweep(scale=args.scale, dry_run=is_dry_run)

    if args.phase in ("all", "main"):
        run_phase2_main(dry_run=is_dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
