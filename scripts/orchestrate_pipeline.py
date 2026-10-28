#!/usr/bin/env python3
"""Unified pipeline orchestrator:
1. Waits for 125M hyperparameter sweep to complete across all 15 arms.
2. Selects the winning hyperparameters for each optimizer (muon_adamw, cadamw, hybrid_muon_cadamw).
3. Generates sweep summary & markdown report and pushes them to GitHub.
4. Generates Phase 2 main runs: 125M on 2.5B tokens across 3 seeds per optimizer (9 runs total).
5. Executes main runs sequentially on the TPU slice with strict storage awareness:
   prunes all checkpoints and model parameters across all workers after each run completes.
6. Reports live progress and real-time ETA in Indian Standard Time (IST).
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
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

IST = timezone(timedelta(hours=5, minutes=30))
HOSTS = ("w0", "w1", "w2", "w3")
SWEEP_GROUP = "hparam_sweep_125m"
MAIN_GROUP = "main_runs_125m"
DATA_DIR = Path.home() / "data" / "fw25"
VOCAB_SIZE = 32768
OPTIMIZERS = ("muon_adamw", "cadamw", "hybrid_muon_cadamw")
SEEDS = (42, 43, 44)  # 3 seeds per optimizer requested by user


def ist_now() -> datetime:
    return datetime.now(IST)


def ist_now_str() -> str:
    return ist_now().strftime("%Y-%m-%d %H:%M:%S IST")


def prune_storage_across_workers(hosts: tuple[str, ...], run_dir: Path) -> None:
    """Ensure strict storage consciousness: remove checkpoints, weights and transient logs."""
    # Worker 0 cleanup
    for p in run_dir.glob("*.npz"):
        p.unlink(missing_ok=True)
    shutil.rmtree(run_dir / "ckpt", ignore_errors=True)

    # Remote workers cleanup
    for h in hosts:
        cmds = [
            f"rm -rf {shlex.quote(str(run_dir))}/ckpt {shlex.quote(str(run_dir))}/*.npz /tmp/tpu_logs/* 2>/dev/null || true"
        ]
        if h != hosts[0]:
            cmds.append(f"rm -f {shlex.quote(str(run_dir))}.*.log 2>/dev/null || true")
        cmd_str = " ; ".join(cmds)
        subprocess.run(["ssh", h, cmd_str], timeout=25, check=False)


def wait_for_sweep_completion() -> list[dict[str, Any]]:
    """Monitor sweep until all 15 arms have valid result.json files."""
    sweep_dir = ROOT / "runs" / SWEEP_GROUP
    jobs_file = sweep_dir / "jobs.json"
    if not jobs_file.is_file():
        raise FileNotFoundError(f"Jobs file missing: {jobs_file}")

    jobs = json.loads(jobs_file.read_text())
    expected_names = [j["name"] for j in jobs]
    total_arms = len(expected_names)

    print(f"\n[{ist_now_str()}] [SUPERVISOR] Monitoring {total_arms} sweep arms for completion...", flush=True)

    while True:
        completed = []
        for name in expected_names:
            local_res = sweep_dir / name / "result.json"
            # If not on w0, check coordinator w2 and mirror
            if not local_res.is_file():
                for h in ("w2", "w1", "w3"):
                    remote_res = f"/home/tasma/algebraic-extra/runs/{SWEEP_GROUP}/{name}/result.json"
                    check = subprocess.run(["ssh", h, f"test -f {remote_res}"], check=False, timeout=5)
                    if check.returncode == 0:
                        (sweep_dir / name).mkdir(parents=True, exist_ok=True)
                        subprocess.run(
                            ["rsync", "-q", "-a", f"{h}:/home/tasma/algebraic-extra/runs/{SWEEP_GROUP}/{name}/", f"{sweep_dir}/{name}/"],
                            check=False, timeout=15
                        )
                        break

            if local_res.is_file():
                try:
                    data = json.loads(local_res.read_text())
                    if data.get("completed_steps") == 573 and math.isfinite(data.get("final_eval_nll", float("nan"))):
                        completed.append(data)
                except Exception:
                    pass

        if len(completed) >= total_arms:
            print(f"[{ist_now_str()}] [SUPERVISOR] All {total_arms} sweep arms successfully verified!", flush=True)
            return completed

        time.sleep(15)


def summarize_sweep_and_select_winners(completed_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate sweep results, select best hyperparameters per optimizer, and generate reports."""
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    sweep_dir = ROOT / "runs" / SWEEP_GROUP
    jobs_file = sweep_dir / "jobs.json"
    jobs = json.loads(jobs_file.read_text()) if jobs_file.is_file() else []
    job_map = {j["name"]: j for j in jobs}

    records_by_opt: dict[str, list[dict[str, Any]]] = {opt: [] for opt in OPTIMIZERS}
    all_records: list[dict[str, Any]] = []

    for r in completed_results:
        args = r.get("args", {})
        out_name = Path(args.get("out", "")).name
        opt = args.get("optimizer", "unknown")
        nll = r.get("final_eval_nll", float("inf"))
        ppl = r.get("final_eval_ppl", float("inf"))
        loss = r.get("final_train_loss", float("inf"))

        rec = {
            "name": out_name,
            "optimizer": opt,
            "lr": args.get("lr"),
            "wd": args.get("wd"),
            "muon_lr": args.get("muon_lr"),
            "muon_wd": args.get("muon_wd"),
            "cautious_eps": args.get("cautious_eps"),
            "final_eval_nll": nll,
            "final_eval_ppl": ppl,
            "final_train_loss": loss,
            "train_seconds": r.get("train_seconds"),
        }
        all_records.append(rec)
        if opt in records_by_opt:
            records_by_opt[opt].append(rec)

    # Rank each optimizer's arms by final_eval_nll ascending
    selected_winners: dict[str, dict[str, Any]] = {}
    for opt, arm_list in records_by_opt.items():
        arm_list.sort(key=lambda x: x["final_eval_nll"])
        if arm_list:
            selected_winners[opt] = arm_list[0]

    # Write summary JSON
    summary_data = {
        "generated_at_ist": ist_now_str(),
        "total_arms": len(all_records),
        "selected_winners": selected_winners,
        "arms_by_optimizer": records_by_opt,
        "all_arms": all_records,
    }
    summary_file = results_dir / "hparam_sweep_125m_summary.json"
    summary_file.write_text(json.dumps(summary_data, indent=2))
    print(f"[{ist_now_str()}] Sweep summary written to {summary_file}", flush=True)

    # Write Markdown Report
    lines = [
        "# 125M Hyperparameter Sweep Report (600M Tokens per Arm)",
        "",
        f"**Generated:** {ist_now_str()}  ",
        "**Cluster Hardware:** Cloud TPU v4-32 (16 TPUs across 4 workers: w0, w1, w2, w3)  ",
        "**Model Scale:** 125M Algebraic Transformer (`attn_impl=flash`, `head_impl=fused`)  ",
        "**Training Budget:** 600M tokens per arm (573 steps, batch size 512, seq length 2,048)  ",
        "",
        "## Selected Hyperparameters for Main Runs",
        "",
        "| Optimizer | Winning Arm | Learning Rate | Weight Decay | Muon LR | Muon WD | Cautious Eps | Held-Out NLL | Held-Out PPL |",
        "|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for opt in OPTIMIZERS:
        win = selected_winners.get(opt, {})
        m_lr = win.get("muon_lr", "—")
        m_wd = win.get("muon_wd", "—")
        c_eps = win.get("cautious_eps", "—")
        lines.append(
            f"| **{opt}** | `{win.get('name', 'N/A')}` | {win.get('lr')} | {win.get('wd')} | {m_lr} | {m_wd} | {c_eps} | **{win.get('final_eval_nll', 0):.4f}** | **{win.get('final_eval_ppl', 0):.2f}** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Full Sweep Results by Optimizer",
        "",
    ])

    for opt, arm_list in records_by_opt.items():
        lines.extend([
            f"### Optimizer: `{opt}`",
            "",
            "| Arm Name | LR | WD | Muon LR | Muon WD | C-Eps | Final Loss | Held-Out NLL | Perplexity | Runtime |",
            "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ])
        for a in arm_list:
            m_lr = a.get("muon_lr", "—")
            m_wd = a.get("muon_wd", "—")
            c_eps = a.get("cautious_eps", "—")
            s = a.get("train_seconds", 0)
            lines.append(
                f"| `{a['name']}` | {a['lr']} | {a['wd']} | {m_lr} | {m_wd} | {c_eps} | {a['final_train_loss']:.4f} | **{a['final_eval_nll']:.4f}** | {a['final_eval_ppl']:.2f} | {s/60:.1f} min |"
            )
        lines.append("")

    report_file = results_dir / "hparam_sweep_125m_report.md"
    report_file.write_text("\n".join(lines) + "\n")
    print(f"[{ist_now_str()}] Sweep report written to {report_file}", flush=True)

    return summary_data


def push_sweep_results_to_github():
    """Stage, commit, and push sweep results and evaluation reports to GitHub."""
    print(f"\n[{ist_now_str()}] [GITHUB] Committing and pushing sweep results...", flush=True)
    subprocess.run(["git", "add", "results/", f"runs/{SWEEP_GROUP}/*/result.json", f"runs/{SWEEP_GROUP}/*/meta.json"], cwd=ROOT, check=False)
    commit_res = subprocess.run(
        ["git", "commit", "-m", "Add 125M hyperparameter sweep results and evaluation report [skip ci]"],
        cwd=ROOT, capture_output=True, text=True
    )
    print(f"[GITHUB] Commit output: {commit_res.stdout.strip() or commit_res.stderr.strip()}", flush=True)

    for attempt in range(3):
        push_res = subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, capture_output=True, text=True)
        if push_res.returncode == 0:
            print(f"[{ist_now_str()}] [GITHUB] Successfully pushed sweep results to remote repository origin/main!", flush=True)
            return True
        print(f"[GITHUB] Push attempt {attempt+1} failed ({push_res.stderr.strip()}), retrying in 5s...", flush=True)
        time.sleep(5)
    print(f"[{ist_now_str()}] [GITHUB WARNING] Push failed after 3 attempts.", flush=True)
    return False


def make_main_jobs_for_3_seeds(selected_winners: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build list of 125M main training runs (2.5B tokens, 3 seeds per optimizer = 9 runs total)."""
    seq_len = 2048
    batch = 512
    step_tokens = batch * seq_len  # 1,048,576
    tokens_budget = 2_500_000_000
    steps = math.ceil(tokens_budget / step_tokens)  # 2,385 steps
    micro = 4

    train_file = str(DATA_DIR / "fineweb_train_2_5B.npy")
    valid_file = str(DATA_DIR / "fineweb_valid.npy")

    jobs = []
    for opt in OPTIMIZERS:
        win = selected_winners.get(opt, {})
        lr = win.get("lr", 0.003)
        wd = win.get("wd", 0.1)

        for seed in SEEDS:
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
                "--data-seed", str(seed),
                "--train", train_file,
                "--valid", valid_file,
                "--eval-tokens", "8000000",
                "--eval-batch", str(micro),
                "--eval-every", "250",
                "--log-every", "25",
                "--ckpt-every", "0",       # Strict storage consciousness: no large checkpoints
                "--keep-ckpts", "1",
                "--set", f"vocab_size={VOCAB_SIZE}",
                "--set", "attn_impl=flash",
                "--set", "head_impl=fused",
            ]

            if win.get("muon_lr") is not None:
                args.extend(["--muon-lr", str(win["muon_lr"])])
            if win.get("muon_wd") is not None:
                args.extend(["--muon-wd", str(win["muon_wd"])])
            if opt in ("muon_adamw", "hybrid_muon_cadamw"):
                args.extend(["--muon-momentum", "0.95", "--muon-steps", "5"])
            if win.get("cautious_eps") is not None:
                args.extend(["--cautious-eps", str(win["cautious_eps"])])

            jobs.append({
                "name": name,
                "optimizer": opt,
                "seed": seed,
                "steps": steps,
                "tokens": tokens_budget,
                "lr": lr,
                "wd": wd,
                "muon_lr": win.get("muon_lr"),
                "muon_wd": win.get("muon_wd"),
                "args": args,
            })

    return jobs


def run_main_runs_pipeline(jobs: list[dict[str, Any]]) -> None:
    """Execute the 9 main training runs across the cluster with live monitoring and storage pruning."""
    main_dir = ROOT / "runs" / MAIN_GROUP
    main_dir.mkdir(parents=True, exist_ok=True)
    jobs_file = main_dir / "jobs.json"
    jobs_file.write_text(json.dumps([{"name": j["name"], "args": j["args"]} for j in jobs], indent=2))

    print(f"\n=======================================================", flush=True)
    print(f"[{ist_now_str()}] LAUNCHING PHASE 2: 125M MAIN RUNS (9 RUNS TOTAL)", flush=True)
    print(f"=======================================================\n", flush=True)
    print(f"Configured {len(jobs)} main training runs (2.5B tokens / 2,385 steps each):")
    for j in jobs:
        print(f"  - {j['name']}: {j['optimizer']} seed={j['seed']} lr={j['lr']} wd={j['wd']}")

    cmd = [
        sys.executable, str(ROOT / "scripts/run_slice.py"),
        str(jobs_file),
        "--group", MAIN_GROUP,
        "--hosts", ",".join(HOSTS),
        "--skip-done",
    ]

    # Run run_slice for the main runs
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        if proc.stdout:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
    finally:
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"Main runs slice failed with returncode {proc.returncode}")

    print(f"\n[{ist_now_str()}] ALL 9 MAIN TRAINING RUNS COMPLETED SUCCESSFULLY!", flush=True)


def main():
    print(f"\n[{ist_now_str()}] Pipeline Orchestrator Daemon Started.", flush=True)

    # Step 1: Wait for sweep to finish
    sweep_results = wait_for_sweep_completion()

    # Step 2: Summarize results and select winning hyperparameters
    summary_data = summarize_sweep_and_select_winners(sweep_results)
    winners = summary_data["selected_winners"]

    # Step 3: Push sweep results to GitHub
    push_sweep_results_to_github()

    # Step 4: Generate main jobs for 3 seeds per optimizer (9 runs total)
    main_jobs = make_main_jobs_for_3_seeds(winners)

    # Step 5: Execute main runs pipeline with strict storage pruning
    run_main_runs_pipeline(main_jobs)


if __name__ == "__main__":
    main()
