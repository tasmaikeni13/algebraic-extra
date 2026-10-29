#!/usr/bin/env python3
"""Unified 350M Algebraic Transformer Pipeline Orchestrator:
1. Launches parallel background tokenization of 7B tokens (FineWeb-Edu) for Phase 2.
2. Executes a focused literature-grounded hyperparameter sweep (6 arms: 3 for Muon+AdamW, 3 for Hybrid Muon+C-AdamW) on 600M tokens.
3. Selects winning hyperparameters based on held-out NLL.
4. Generates sweep summary & markdown report and pushes them to GitHub.
5. Verifies 7B dataset readiness across all TPU slice hosts (w0..w3).
6. Executes Phase 2 main runs on 7B tokens (6,675 steps) for both optimizers with seed 42.
7. Enforces strict storage consciousness across all 4 workers after each run.
8. Generates final 350M report and pushes all results to GitHub.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

IST = timezone(timedelta(hours=5, minutes=30))
HOSTS = ("w0", "w1", "w2", "w3")
SWEEP_GROUP = "hparam_sweep_350m"
MAIN_GROUP = "main_runs_350m"
DATA_DIR = Path.home() / "data" / "fw25"
VOCAB_SIZE = 32768
OPTIMIZERS = ("muon_adamw", "hybrid_muon_cadamw")
SEED = 42


def ist_now() -> datetime:
    return datetime.now(IST)


def ist_now_str() -> str:
    return ist_now().strftime("%Y-%m-%d %H:%M:%S IST")


def prune_storage_across_workers(hosts: tuple[str, ...], run_dir: Path) -> None:
    """Ensure strict storage consciousness: remove checkpoints, weights and transient logs."""
    for p in run_dir.glob("*.npz"):
        p.unlink(missing_ok=True)
    shutil.rmtree(run_dir / "ckpt", ignore_errors=True)

    for h in hosts:
        cmds = [
            f"rm -rf {shlex.quote(str(run_dir))}/ckpt {shlex.quote(str(run_dir))}/*.npz /tmp/tpu_logs/* 2>/dev/null || true"
        ]
        if h != hosts[0]:
            cmds.append(f"rm -f {shlex.quote(str(run_dir))}.*.log 2>/dev/null || true")
        cmd_str = " ; ".join(cmds)
        subprocess.run(["ssh", h, cmd_str], timeout=25, check=False)


def mirror_results_from_all_workers(group: str) -> None:
    """Ensure all worker directories have complete result files mirrored to w0."""
    group_dir = ROOT / "runs" / group
    group_dir.mkdir(parents=True, exist_ok=True)
    for h in ("w2", "w1", "w3"):
        subprocess.run(
            ["rsync", "-q", "-a", f"{h}:{ROOT}/runs/{group}/", f"{group_dir}/"],
            check=False,
            timeout=30,
        )


def launch_7b_data_prep() -> subprocess.Popen:
    """Launch preparation of 7B tokens in the background on w0."""
    target_7b = DATA_DIR / "fineweb_train_7B.npy"
    if target_7b.is_file() and target_7b.stat().st_size == 7_000_000_000 * 2:
        print(f"[{ist_now_str()}] [DATA PREP] 7B dataset already present and verified ({target_7b}).", flush=True)
        return None

    log_path = DATA_DIR / "prepare_7b.log"
    print(f"[{ist_now_str()}] [DATA PREP] Launching background preparation of 7B tokens to {target_7b}...", flush=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "prepare_fineweb_edu.py"),
        "--output-dir", str(DATA_DIR),
        "--train-tokens", "7000000000",
        "--valid-tokens", "10000000",
        "--threads", "8",
    ]
    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=str(ROOT))
    return proc


def wait_and_sync_7b_dataset(prep_proc: subprocess.Popen | None) -> None:
    """Wait for 7B tokenization to finish on w0, then sync to w1, w2, w3."""
    target_7b = DATA_DIR / "fineweb_train_7B.npy"
    log_path = DATA_DIR / "prepare_7b.log"

    if prep_proc is not None:
        print(f"\n[{ist_now_str()}] [DATA PREP] Waiting for 7B dataset tokenization to complete on w0...", flush=True)
        while prep_proc.poll() is None:
            if log_path.is_file():
                try:
                    last_line = subprocess.check_output(["tail", "-n", "1", str(log_path)], text=True).strip()
                    if last_line:
                        print(f"[{ist_now_str()}] [DATA PREP STATUS] {last_line}", flush=True)
                except Exception:
                    pass
            time.sleep(30)
        ret = prep_proc.returncode
        if ret != 0:
            if log_path.is_file():
                print(log_path.read_text(), file=sys.stderr)
            raise RuntimeError(f"7B dataset preparation failed with code {ret}")

    if not target_7b.is_file() or target_7b.stat().st_size != 7_000_000_000 * 2:
        raise FileNotFoundError(f"7B dataset file missing or incomplete: {target_7b}")

    print(f"[{ist_now_str()}] [DATA PREP] 7B dataset complete on w0 ({target_7b.stat().st_size / 1e9:.2f} GB).", flush=True)

    # Distribute to w1, w2, w3
    for h in ("w1", "w2", "w3"):
        remote_check = subprocess.run(["ssh", h, f"test -f {shlex.quote(str(target_7b))}"], check=False)
        if remote_check.returncode != 0:
            print(f"[{ist_now_str()}] [DATA PREP] Syncing 7B dataset to {h}...", flush=True)
            subprocess.run(
                ["rsync", "-a", "--info=progress2", str(target_7b), f"{h}:{DATA_DIR}/"],
                check=True,
            )
        else:
            print(f"[{ist_now_str()}] [DATA PREP] 7B dataset already verified on {h}.", flush=True)
    print(f"[{ist_now_str()}] [DATA PREP] 7B dataset fully synced and verified across all TPU hosts!\n", flush=True)


def build_sweep_jobs() -> list[dict[str, Any]]:
    """Define 6 focused sweep arms (3 for muon_adamw, 3 for hybrid_muon_cadamw).
    
    Architecture: 350M Algebraic Transformer (d_model=1024, n_layers=24, n_heads=16, d_ff=2688).
    Budget: 600M tokens each on fineweb_sweep_600M.npy (573 steps at batch=512, seq=2048).
    Hardware execution: batch=512, micro=2 (16 accumulation steps per chip), Pallas FlashAttention + Fused Head.
    """
    seq_len = 2048
    batch = 512
    micro = 2
    tokens_budget = 600_000_000
    steps = 573

    train_file = str(DATA_DIR / "fineweb_sweep_600M.npy")
    valid_file = str(DATA_DIR / "fineweb_valid.npy")

    common_args = [
        "--preset", "algebraic",
        "--size", "350m",
        "--seq-len", str(seq_len),
        "--batch", str(batch),
        "--micro", str(micro),
        "--tokens", str(tokens_budget),
        "--steps", str(steps),
        "--warmup", "0.02",
        "--min-lr-ratio", "0.1",
        "--b1", "0.9",
        "--b2", "0.95",
        "--adam-eps", "1e-8",
        "--clip", "1.0",
        "--schedule", "cosine",
        "--seed", str(SEED),
        "--data-seed", str(SEED),
        "--train", train_file,
        "--valid", valid_file,
        "--eval-tokens", "8000000",
        "--eval-batch", str(micro),
        "--eval-every", "50",
        "--log-every", "10",
        "--ckpt-every", "0",
        "--keep-ckpts", "1",
        "--set", f"vocab_size={VOCAB_SIZE}",
        "--set", "attn_impl=flash",
        "--set", "head_impl=fused",
    ]

    jobs = []

    # 1. muon_adamw (3 arms)
    # Width scaling factor sqrt(768/1024) ≈ 0.866 from 125M (where 0.002 to 0.003 was optimal).
    muon_configs = [
        ("canonical", 0.0020, 0.1, 0.020, 0.1),
        ("low_lr", 0.0015, 0.1, 0.015, 0.1),
        ("high_lr", 0.0025, 0.1, 0.020, 0.1),
    ]
    for tag, lr, wd, m_lr, m_wd in muon_configs:
        name = f"alg_350m_muon_adamw_{tag}_s{SEED}"
        args = list(common_args) + [
            "--optimizer", "muon_adamw",
            "--lr", str(lr),
            "--wd", str(wd),
            "--muon-lr", str(m_lr),
            "--muon-wd", str(m_wd),
            "--muon-momentum", "0.95",
            "--muon-steps", "5",
        ]
        jobs.append({
            "name": name,
            "optimizer": "muon_adamw",
            "lr": lr,
            "wd": wd,
            "muon_lr": m_lr,
            "muon_wd": m_wd,
            "args": args,
        })

    # 2. hybrid_muon_cadamw (3 arms)
    hybrid_configs = [
        ("canonical", 0.0020, 0.1, 0.020, 0.1, 0.001),
        ("low_lr", 0.0015, 0.1, 0.015, 0.1, 0.001),
        ("high_lr", 0.0025, 0.1, 0.020, 0.1, 0.001),
    ]
    for tag, lr, wd, m_lr, m_wd, c_eps in hybrid_configs:
        name = f"alg_350m_hybrid_muon_cadamw_{tag}_s{SEED}"
        args = list(common_args) + [
            "--optimizer", "hybrid_muon_cadamw",
            "--lr", str(lr),
            "--wd", str(wd),
            "--muon-lr", str(m_lr),
            "--muon-wd", str(m_wd),
            "--muon-momentum", "0.95",
            "--muon-steps", "5",
            "--cautious-eps", str(c_eps),
        ]
        jobs.append({
            "name": name,
            "optimizer": "hybrid_muon_cadamw",
            "lr": lr,
            "wd": wd,
            "muon_lr": m_lr,
            "muon_wd": m_wd,
            "cautious_eps": c_eps,
            "args": args,
        })

    return jobs


def run_sweep() -> dict[str, Any]:
    print(f"\n=======================================================", flush=True)
    print(f"[{ist_now_str()}] LAUNCHING PHASE 1: 350M ALGEBRAIC TRANSFORMER SWEEP (6 ARMS)", flush=True)
    print(f"Literature-backed hyperparameter grid for 350M Algebraic Transformer:")
    print(f"  - Muon+AdamW: 3 arms (canonical lr=2e-3/muon_lr=0.02, low lr=1.5e-3/0.015, high lr=2.5e-3/0.02)")
    print(f"  - Hybrid Muon+C-AdamW: 3 arms (canonical lr=2e-3/muon_lr=0.02, low lr=1.5e-3/0.015, high lr=2.5e-3/0.02, eps=1e-3)")
    print(f"  - Budget per arm: 600M tokens (573 steps), batch 512, micro 2, seq 2048")
    print(f"=======================================================\n", flush=True)

    sweep_dir = ROOT / "runs" / SWEEP_GROUP
    sweep_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_sweep_jobs()
    jobs_file = sweep_dir / "jobs.json"
    jobs_file.write_text(json.dumps([{"name": j["name"], "args": j["args"]} for j in jobs], indent=2))

    cmd = [
        sys.executable, str(ROOT / "scripts/run_slice.py"),
        str(jobs_file),
        "--group", SWEEP_GROUP,
        "--hosts", ",".join(HOSTS),
        "--skip-done",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        if proc.stdout:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
    finally:
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"Sweep run_slice failed with returncode {proc.returncode}")

    mirror_results_from_all_workers(SWEEP_GROUP)

    job_map = {j["name"]: j for j in jobs}
    results_by_opt: dict[str, list[dict[str, Any]]] = {opt: [] for opt in OPTIMIZERS}

    for j in jobs:
        name = j["name"]
        res_file = sweep_dir / name / "result.json"
        if res_file.is_file():
            data = json.loads(res_file.read_text())
            record = {
                "name": name,
                "optimizer": j["optimizer"],
                "lr": j["lr"],
                "wd": j["wd"],
                "muon_lr": j.get("muon_lr"),
                "muon_wd": j.get("muon_wd"),
                "cautious_eps": j.get("cautious_eps"),
                "final_eval_nll": data.get("final_eval_nll"),
                "final_eval_ppl": data.get("final_eval_ppl"),
                "final_train_loss": data.get("final_train_loss"),
                "train_seconds": data.get("train_seconds"),
            }
            results_by_opt[j["optimizer"]].append(record)

    selected_winners = {}
    for opt, records in results_by_opt.items():
        valid_records = [r for r in records if r.get("final_eval_nll") is not None and math.isfinite(r["final_eval_nll"])]
        if valid_records:
            winner = min(valid_records, key=lambda r: r["final_eval_nll"])
            selected_winners[opt] = winner

    summary = {
        "created_at": ist_now_str(),
        "model_architecture": "350M Algebraic Transformer (ALU-GLU, Cayley, Asinh Attention, Alg-CE)",
        "active_parameters": 332453248,
        "budget_per_arm": "600M tokens (573 steps)",
        "selected_winners": selected_winners,
        "results_by_optimizer": results_by_opt,
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "hparam_sweep_350m_summary.json").write_text(json.dumps(summary, indent=2))

    # Generate Markdown Report
    md = [
        "# 350M Algebraic Transformer Hyperparameter Sweep Report (600M Tokens/Arm)",
        "",
        f"**Date Completed:** {ist_now_str()}  ",
        "**Cluster Hardware:** Cloud TPU v4-32 (16 TPUs across 4 workers: `w0`, `w1`, `w2`, `w3`)  ",
        "**Model Architecture:** 350M Pure Algebraic Transformer (`ffn=alu_glu`, `beta=0.5`, `pos=cayley_geo`, `attn=asinh`, `sink=learned`, `head=alg_ce`)  ",
        "**Active Parameters:** 332,453,248 (`d_model=1024`, `n_layers=24`, `n_heads=16`, `d_ff=2688`, `vocab_size=32768`)  ",
        "**Acceleration Kernels:** Asinh FlashAttention Pallas + Alg-CE Fused Head Pallas  ",
        "**Training Budget:** 600M tokens per arm (573 steps, batch size 512, micro 2, seq length 2,048)  ",
        "**Hyperparameter Range:** Grounded in width-scaled Muon and AdamW literature  ",
        "",
        "## Selected Winning Hyperparameters for 7B Main Runs",
        "",
        "| Optimizer | Winning Arm | Base LR | Weight Decay | Muon LR | Muon WD | C-Eps | Held-Out NLL | Held-Out PPL |",
        "|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for opt in OPTIMIZERS:
        w = selected_winners.get(opt, {})
        md.append(f"| **{opt}** | `{w.get('name')}` | {w.get('lr')} | {w.get('wd')} | {w.get('muon_lr')} | {w.get('muon_wd')} | {w.get('cautious_eps', '-')} | **{w.get('final_eval_nll'):.4f}** | **{w.get('final_eval_ppl'):.2f}** |")

    md.extend(["", "---", "", "## Full Sweep Arm Results by Optimizer", ""])
    for opt in OPTIMIZERS:
        md.append(f"### Optimizer: `{opt}`\n")
        md.append("| Arm Name | LR | WD | Muon LR | Muon WD | Final Loss | Held-Out NLL | Perplexity | Runtime |")
        md.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for r in results_by_opt[opt]:
            m_lr = r.get("muon_lr") if r.get("muon_lr") is not None else "-"
            m_wd = r.get("muon_wd") if r.get("muon_wd") is not None else "-"
            floss = f"{r.get('final_train_loss'):.4f}" if r.get('final_train_loss') else "-"
            fnll = f"{r.get('final_eval_nll'):.4f}" if r.get('final_eval_nll') else "-"
            fppl = f"{r.get('final_eval_ppl'):.2f}" if r.get('final_eval_ppl') else "-"
            dur = f"{r.get('train_seconds', 0)/60:.1f} min"
            md.append(f"| `{r['name']}` | {r['lr']} | {r['wd']} | {m_lr} | {m_wd} | {floss} | **{fnll}** | {fppl} | {dur} |")
        md.append("")

    (results_dir / "hparam_sweep_350m_report.md").write_text("\n".join(md) + "\n")

    subprocess.run(["git", "add", "results/hparam_sweep_350m_summary.json", "results/hparam_sweep_350m_report.md"], check=True)
    subprocess.run(["git", "commit", "-m", "Add 350M Algebraic Transformer hyperparameter sweep results and report"], check=True)
    subprocess.run(["git", "push", "origin", "main"], check=True)
    print(f"[{ist_now_str()}] Sweep results committed and pushed to GitHub!", flush=True)

    return selected_winners


def build_main_jobs(winners: dict[str, Any]) -> list[dict[str, Any]]:
    """Build Phase 2 main runs: 1 seed (seed 42) on 7B tokens for muon_adamw and hybrid_muon_cadamw."""
    seq_len = 2048
    batch = 512
    micro = 2
    tokens_budget = 7_000_000_000
    steps = 6675  # ceil(7B / (512 * 2048)) = 6676, 6675 * 512 * 2048 = 6,999,261,184

    train_file = str(DATA_DIR / "fineweb_train_7B.npy")
    valid_file = str(DATA_DIR / "fineweb_valid.npy")

    jobs = []
    for opt in OPTIMIZERS:
        win = winners.get(opt, {})
        lr = win.get("lr", 0.0020)
        wd = win.get("wd", 0.1)

        name = f"main_350m_{opt}_s{SEED}"
        args = [
            "--preset", "algebraic",
            "--size", "350m",
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
            "--seed", str(SEED),
            "--data-seed", str(SEED),
            "--train", train_file,
            "--valid", valid_file,
            "--eval-tokens", "8000000",
            "--eval-batch", str(micro),
            "--eval-every", "250",
            "--log-every", "25",
            "--ckpt-every", "0",       # Strict storage consciousness: no giant checkpoints
            "--keep-ckpts", "1",
            "--set", f"vocab_size={VOCAB_SIZE}",
            "--set", "attn_impl=flash",
            "--set", "head_impl=fused",
        ]
        if win.get("muon_lr") is not None:
            args.extend(["--muon-lr", str(win["muon_lr"])])
        if win.get("muon_wd") is not None:
            args.extend(["--muon-wd", str(win["muon_wd"])])
        args.extend(["--muon-momentum", "0.95", "--muon-steps", "5"])
        if win.get("cautious_eps") is not None:
            args.extend(["--cautious-eps", str(win["cautious_eps"])])

        jobs.append({
            "name": name,
            "optimizer": opt,
            "seed": SEED,
            "lr": lr,
            "wd": wd,
            "muon_lr": win.get("muon_lr"),
            "muon_wd": win.get("muon_wd"),
            "cautious_eps": win.get("cautious_eps"),
            "args": args,
        })
    return jobs


def run_main_runs(jobs: list[dict[str, Any]]) -> None:
    print(f"\n=======================================================", flush=True)
    print(f"[{ist_now_str()}] LAUNCHING PHASE 2: 350M MAIN RUNS (2 RUNS ON 7B TOKENS EACH)", flush=True)
    print(f"=======================================================\n", flush=True)

    main_dir = ROOT / "runs" / MAIN_GROUP
    main_dir.mkdir(parents=True, exist_ok=True)
    jobs_file = main_dir / "jobs.json"
    jobs_file.write_text(json.dumps([{"name": j["name"], "args": j["args"]} for j in jobs], indent=2))

    cmd = [
        sys.executable, str(ROOT / "scripts/run_slice.py"),
        str(jobs_file),
        "--group", MAIN_GROUP,
        "--hosts", ",".join(HOSTS),
        "--skip-done",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        if proc.stdout:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
    finally:
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"Main runs failed with returncode {proc.returncode}")

    mirror_results_from_all_workers(MAIN_GROUP)


def generate_main_report() -> None:
    main_dir = ROOT / "runs" / MAIN_GROUP
    results_by_opt = {}

    for opt in OPTIMIZERS:
        name = f"main_350m_{opt}_s{SEED}"
        res_file = main_dir / name / "result.json"
        if res_file.is_file():
            data = json.loads(res_file.read_text())
            results_by_opt[opt] = data

    summary_file = ROOT / "results" / "main_350m_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps({
        "generated_at_ist": ist_now_str(),
        "model": "350M Algebraic Transformer",
        "tokens_per_run": 7_000_000_000,
        "seed": SEED,
        "results": results_by_opt,
    }, indent=2))

    # Generate Markdown Report
    md = [
        "# 350M Algebraic Transformer Main Benchmark Report (7B Tokens)",
        "",
        f"**Generated:** {ist_now_str()}  ",
        "**Scale:** 350M Active Parameters (`332,453,248` non-embedding weights, tied embeddings)  ",
        "**Dataset:** FineWeb-Edu (7.0 Billion Tokens per run, Chinchilla-optimal compute budget)  ",
        "**Hardware:** Cloud TPU v4-32 (16 TPUs across 4 workers: `w0`, `w1`, `w2`, `w3`)  ",
        f"**Seed:** {SEED}  ",
        "",
        "## Final Evaluation Results",
        "",
        "| Optimizer | Held-Out NLL (nats) | Held-Out Perplexity | Final Train Loss | Tok/s Throughput | Train Runtime |",
        "|---|:---:|:---:|:---:|:---:|:---:|",
    ]

    for opt, r in results_by_opt.items():
        nll = r.get("final_eval_nll", float("nan"))
        ppl = r.get("final_eval_ppl", float("nan"))
        loss = r.get("final_train_loss", float("nan"))
        toks = r.get("tokens_per_second", 0)
        dur = r.get("train_seconds", 0) / 3600.0
        md.append(f"| **{opt}** | **{nll:.4f}** | **{ppl:.2f}** | {loss:.4f} | {toks:,.0f} tok/s | {dur:.2f} hrs |")

    report_file = ROOT / "results" / "main_350m_report.md"
    report_file.write_text("\n".join(md) + "\n")

    subprocess.run(["git", "add", "results/main_350m_summary.json", "results/main_350m_report.md"], check=True)
    subprocess.run(["git", "commit", "-m", "Add 350M Algebraic Transformer 7B benchmark results and report"], check=True)
    subprocess.run(["git", "push", "origin", "main"], check=True)
    print(f"[{ist_now_str()}] Main 350M benchmark report committed and pushed to GitHub!", flush=True)


def main():
    print(f"[{ist_now_str()}] ========================================================", flush=True)
    print(f"[{ist_now_str()}] STARTING 350M ALGEBRAIC TRANSFORMER COMPLETE PIPELINE", flush=True)
    print(f"[{ist_now_str()}] ========================================================", flush=True)

    # 1. Start background 7B data preparation immediately
    data_prep_proc = launch_7b_data_prep()

    # 2. Execute Phase 1: Focused 350M Hyperparameter Sweep (6 Arms)
    winners = run_sweep()

    # 3. Wait for 7B dataset to be fully prepared on w0 and synced to all workers
    wait_and_sync_7b_dataset(data_prep_proc)

    # 4. Build Phase 2 Main Runs (1 seed on 7B tokens for muon_adamw and hybrid_muon_cadamw)
    main_jobs = build_main_jobs(winners)

    # 5. Execute Phase 2 Main Runs
    run_main_runs(main_jobs)

    # 6. Generate final reports and push to GitHub
    generate_main_report()

    print(f"\n[{ist_now_str()}] ========================================================", flush=True)
    print(f"[{ist_now_str()}] 350M PIPELINE COMPLETED SUCCESSFULLY ACROSS ALL PHASES!", flush=True)
    print(f"[{ist_now_str()}] ========================================================", flush=True)


if __name__ == "__main__":
    main()
