#!/usr/bin/env python3
"""Run 125M Algebraic Transformer training for seeds 45 and 46 across all 3 optimizers.

Executes 6 runs (2.5B tokens / 2,385 steps each):
- main_125m_muon_adamw_s45, main_125m_muon_adamw_s46
- main_125m_cadamw_s45, main_125m_cadamw_s46
- main_125m_hybrid_muon_cadamw_s45, main_125m_hybrid_muon_cadamw_s46

Strict storage consciousness is enforced: intermediate checkpoints and weights
are pruned immediately across all 4 worker nodes.
Once all 6 runs finish, it mirrors results, regenerates the 5-seed summary & report,
and pushes them to GitHub.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

IST = timezone(timedelta(hours=5, minutes=30))
HOSTS = ("w0", "w1", "w2", "w3")
MAIN_GROUP = "main_runs_125m"
DATA_DIR = Path.home() / "data" / "fw25"
VOCAB_SIZE = 32768
NEW_SEEDS = (45, 46)


def ist_now_str() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def make_jobs() -> list[dict]:
    seq_len = 2048
    batch = 512
    micro = 4
    tokens_budget = 2_500_000_000
    steps = 2385

    train_file = str(DATA_DIR / "fineweb_train_2_5B.npy")
    valid_file = str(DATA_DIR / "fineweb_valid.npy")

    # Winning configurations from 125M hyperparameter sweep
    configs = {
        "muon_adamw": {
            "lr": 0.003,
            "wd": 0.1,
            "muon_lr": 0.02,
            "muon_wd": 0.1,
            "cautious_eps": 0.001,
        },
        "cadamw": {
            "lr": 0.0015,
            "wd": 0.1,
            "muon_lr": None,
            "muon_wd": None,
            "cautious_eps": 0.001,
        },
        "hybrid_muon_cadamw": {
            "lr": 0.003,
            "wd": 0.1,
            "muon_lr": 0.02,
            "muon_wd": 0.1,
            "cautious_eps": 0.001,
        },
    }

    jobs = []
    for opt in ("muon_adamw", "cadamw", "hybrid_muon_cadamw"):
        cfg = configs[opt]
        for seed in NEW_SEEDS:
            name = f"main_125m_{opt}_s{seed}"
            args = [
                "--preset", "algebraic",
                "--size", "125m",
                "--seq-len", str(seq_len),
                "--batch", str(batch),
                "--micro", str(micro),
                "--tokens", str(tokens_budget),
                "--steps", str(steps),
                "--optimizer", opt,
                "--lr", str(cfg["lr"]),
                "--wd", str(cfg["wd"]),
                "--warmup", "0.02",
                "--min-lr-ratio", "0.1",
                "--b1", "0.9",
                "--b2", "0.95",
                "--adam-eps", "1e-8",
                "--clip", "1.0",
                "--schedule", "cosine",
                "--seed", str(seed),
                "--data-seed", str(seed),
                "--train", train_file,
                "--valid", valid_file,
                "--eval-tokens", "8000000",
                "--eval-batch", str(micro),
                "--eval-every", "250",
                "--log-every", "25",
                "--ckpt-every", "0",
                "--keep-ckpts", "1",
                "--set", f"vocab_size={VOCAB_SIZE}",
                "--set", "attn_impl=flash",
                "--set", "head_impl=fused",
            ]
            if cfg["muon_lr"] is not None:
                args.extend(["--muon-lr", str(cfg["muon_lr"])])
            if cfg["muon_wd"] is not None:
                args.extend(["--muon-wd", str(cfg["muon_wd"])])
            if opt in ("muon_adamw", "hybrid_muon_cadamw"):
                args.extend(["--muon-momentum", "0.95", "--muon-steps", "5"])
            if cfg["cautious_eps"] is not None:
                args.extend(["--cautious-eps", str(cfg["cautious_eps"])])

            jobs.append({
                "name": name,
                "args": args,
            })
    return jobs


def main():
    print(f"\n[{ist_now_str()}] Starting 125M Algebraic Transformer runs for seeds 45 & 46 (6 runs total)...", flush=True)

    main_dir = ROOT / "runs" / MAIN_GROUP
    main_dir.mkdir(parents=True, exist_ok=True)

    jobs = make_jobs()
    jobs_file = main_dir / "jobs_seeds_45_46.json"
    jobs_file.write_text(json.dumps(jobs, indent=2))

    print(f"Jobs configured ({len(jobs)} runs):", flush=True)
    for j in jobs:
        print(f"  - {j['name']}", flush=True)

    # Execute runs via run_slice.py
    cmd = [
        sys.executable,
        str(ROOT / "scripts/run_slice.py"),
        str(jobs_file),
        "--group", MAIN_GROUP,
        "--hosts", ",".join(HOSTS),
        "--skip-done",
    ]

    print(f"\n[{ist_now_str()}] Launching run_slice across hosts {HOSTS}...", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        if proc.stdout:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
    finally:
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"run_slice failed with returncode {proc.returncode}")

    print(f"\n[{ist_now_str()}] All 6 runs completed! Mirroring results from coordinator w2...", flush=True)
    subprocess.run(
        ["rsync", "-av", f"w2:{ROOT}/runs/{MAIN_GROUP}/", f"{ROOT}/runs/{MAIN_GROUP}/"],
        check=True,
    )

    print(f"\n[{ist_now_str()}] Regenerating main report across all 5 seeds...", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/generate_main_report.py")], check=True)

    print(f"\n[{ist_now_str()}] Committing and pushing 5-seed benchmark results to GitHub...", flush=True)
    subprocess.run(["git", "add", "results/main_125m_summary.json", "results/main_125m_report.md", "scripts/"], check=True)
    commit_msg = "Update 125M Algebraic Transformer benchmarks to 5 seeds (42, 43, 44, 45, 46)"
    subprocess.run(["git", "commit", "-m", commit_msg], check=True)
    subprocess.run(["git", "push", "origin", "main"], check=True)

    print(f"\n[{ist_now_str()}] ALL 5 SEEDS SUCCESSFULLY BENCHMARKED AND PUSHED TO GITHUB!", flush=True)


if __name__ == "__main__":
    main()
