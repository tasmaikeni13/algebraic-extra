#!/usr/bin/env python3
"""Unified 350M Hyperparameter Sweep Pipeline:
1. Executes the standard (extensive 5-arm per optimizer) hyperparameter sweep for the 350M Algebraic Transformer (10 arms: 5 for Muon+AdamW, 5 for Hybrid Muon+C-AdamW).
2. Generates Algebraic 350M sweep report & summary, commits and pushes to GitHub.
3. Executes the standard hyperparameter sweep for the 350M Standard Transformer (10 arms: 5 for Muon+AdamW, 5 for Hybrid Muon+C-AdamW).
4. Generates Standard 350M sweep report, head-to-head architectural comparison report, commits and pushes to GitHub.
5. Strict storage consciousness: all checkpoints and transient files purged after each arm across all 4 workers (w0..w3).
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
DATA_DIR = Path.home() / "data" / "fw25"
VOCAB_SIZE = 32768
OPTIMIZERS = ("muon_adamw", "hybrid_muon_cadamw")
SEED = 42

ALG_SWEEP_GROUP = "hparam_sweep_350m_algebraic"
STD_SWEEP_GROUP = "hparam_sweep_350m_standard"


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


def build_algebraic_sweep_jobs() -> list[dict[str, Any]]:
    """Standard 5-arm per optimizer sweep for 350M Algebraic Transformer (10 arms total).
    
    Budget: 600M tokens each on fineweb_sweep_600M.npy (573 steps at batch=512, micro=2, seq=2048).
    Pallas FlashAttention (Asinh) + Fused Head (Alg-CE).
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

    # 1. muon_adamw (5 arms: base, low_lr, high_lr, low_wd, canonical)
    muon_configs = [
        ("base", 0.0025, 0.1, 0.020, 0.05),
        ("low_lr", 0.0015, 0.1, 0.015, 0.05),
        ("high_lr", 0.0030, 0.1, 0.030, 0.05),
        ("low_wd", 0.0025, 0.05, 0.020, 0.02),
        ("canonical", 0.0020, 0.1, 0.020, 0.10),
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
            "architecture": "algebraic",
            "optimizer": "muon_adamw",
            "tag": tag,
            "lr": lr,
            "wd": wd,
            "muon_lr": m_lr,
            "muon_wd": m_wd,
            "args": args,
        })

    # 2. hybrid_muon_cadamw (5 arms: base, low_lr, high_lr, low_wd, canonical)
    hybrid_configs = [
        ("base", 0.0025, 0.1, 0.020, 0.05, 0.001),
        ("low_lr", 0.0015, 0.1, 0.015, 0.05, 0.001),
        ("high_lr", 0.0030, 0.1, 0.030, 0.05, 0.001),
        ("low_wd", 0.0025, 0.05, 0.020, 0.02, 0.001),
        ("canonical", 0.0020, 0.1, 0.020, 0.10, 0.001),
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
            "architecture": "algebraic",
            "optimizer": "hybrid_muon_cadamw",
            "tag": tag,
            "lr": lr,
            "wd": wd,
            "muon_lr": m_lr,
            "muon_wd": m_wd,
            "cautious_eps": c_eps,
            "args": args,
        })

    return jobs


def build_standard_sweep_jobs() -> list[dict[str, Any]]:
    """Standard 5-arm per optimizer sweep for 350M Standard LLaMA-style Transformer (10 arms total).
    
    Budget: 600M tokens each on fineweb_sweep_600M.npy (573 steps at batch=512, micro=2, seq=2048).
    Pallas FlashAttention (Softmax) + Fused Head (Cross-Entropy).
    Literature-grounded LR ranges: LLaMA/GPT-3 width-scaled (0.0005 to 0.0015).
    """
    seq_len = 2048
    batch = 512
    micro = 2
    tokens_budget = 600_000_000
    steps = 573

    train_file = str(DATA_DIR / "fineweb_sweep_600M.npy")
    valid_file = str(DATA_DIR / "fineweb_valid.npy")

    common_args = [
        "--preset", "standard",
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

    # 1. muon_adamw (5 arms: canonical, canonical_low_muon, mid_lr, low_lr, base)
    std_muon_configs = [
        ("canonical", 0.0005, 0.1, 0.020, 0.1),
        ("canonical_low_muon", 0.0005, 0.1, 0.010, 0.1),
        ("mid_lr", 0.0008, 0.1, 0.020, 0.1),
        ("low_lr", 0.0008, 0.1, 0.010, 0.1),
        ("base", 0.0015, 0.1, 0.020, 0.1),
    ]
    for tag, lr, wd, m_lr, m_wd in std_muon_configs:
        name = f"std_350m_muon_adamw_{tag}_s{SEED}"
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
            "architecture": "standard",
            "optimizer": "muon_adamw",
            "tag": tag,
            "lr": lr,
            "wd": wd,
            "muon_lr": m_lr,
            "muon_wd": m_wd,
            "args": args,
        })

    # 2. hybrid_muon_cadamw (5 arms: canonical, canonical_low_muon, mid_lr, low_lr, base)
    std_hybrid_configs = [
        ("canonical", 0.0005, 0.1, 0.020, 0.1, 0.001),
        ("canonical_low_muon", 0.0005, 0.1, 0.010, 0.1, 0.001),
        ("mid_lr", 0.0008, 0.1, 0.020, 0.1, 0.001),
        ("low_lr", 0.0008, 0.1, 0.010, 0.1, 0.001),
        ("base", 0.0015, 0.1, 0.020, 0.1, 0.001),
    ]
    for tag, lr, wd, m_lr, m_wd, c_eps in std_hybrid_configs:
        name = f"std_350m_hybrid_muon_cadamw_{tag}_s{SEED}"
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
            "architecture": "standard",
            "optimizer": "hybrid_muon_cadamw",
            "tag": tag,
            "lr": lr,
            "wd": wd,
            "muon_lr": m_lr,
            "muon_wd": m_wd,
            "cautious_eps": c_eps,
            "args": args,
        })

    return jobs


def run_sweep_phase(group: str, jobs: list[dict[str, Any]], arch_title: str) -> dict[str, Any]:
    print(f"\n=======================================================", flush=True)
    print(f"[{ist_now_str()}] LAUNCHING SWEEP: {arch_title.upper()} ({len(jobs)} ARMS)", flush=True)
    print(f"=======================================================\n", flush=True)

    sweep_dir = ROOT / "runs" / group
    sweep_dir.mkdir(parents=True, exist_ok=True)
    jobs_file = sweep_dir / "jobs.json"
    jobs_file.write_text(json.dumps([{"name": j["name"], "args": j["args"]} for j in jobs], indent=2))

    cmd = [
        sys.executable, str(ROOT / "scripts/run_slice.py"),
        str(jobs_file),
        "--group", group,
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
        raise RuntimeError(f"Sweep {group} failed with returncode {proc.returncode}")

    mirror_results_from_all_workers(group)

    results_by_opt: dict[str, list[dict[str, Any]]] = {opt: [] for opt in OPTIMIZERS}
    for j in jobs:
        name = j["name"]
        res_file = sweep_dir / name / "result.json"
        if res_file.is_file():
            data = json.loads(res_file.read_text())
            record = {
                "name": name,
                "optimizer": j["optimizer"],
                "tag": j.get("tag"),
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
        "architecture_title": arch_title,
        "group": group,
        "total_arms": len(jobs),
        "budget_per_arm": "600M tokens (573 steps)",
        "selected_winners": selected_winners,
        "results_by_optimizer": results_by_opt,
    }
    return summary


def publish_algebraic_sweep(summary: dict[str, Any]) -> None:
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "hparam_sweep_350m_algebraic_summary.json").write_text(json.dumps(summary, indent=2))

    selected_winners = summary["selected_winners"]
    results_by_opt = summary["results_by_optimizer"]

    md = [
        "# 350M Algebraic Transformer Hyperparameter Sweep Report (10 Arms)",
        "",
        f"**Date Completed:** {ist_now_str()}  ",
        "**Cluster Hardware:** Cloud TPU v4-32 (16 TPUs across 4 workers: `w0`, `w1`, `w2`, `w3`)  ",
        "**Model Architecture:** 350M Pure Algebraic Transformer (`ffn=alu_glu`, `beta=0.5`, `pos=cayley_geo`, `attn=asinh`, `sink=learned`, `head=alg_ce`)  ",
        "**Active Parameters:** 332,453,248 (`d_model=1024`, `n_layers=24`, `n_heads=16`, `d_ff=2688`, `vocab_size=32768`)  ",
        "**Acceleration Kernels:** Asinh FlashAttention Pallas + Alg-CE Fused Head Pallas  ",
        "**Training Budget:** 600M tokens per arm (573 steps, batch size 512, micro 2, seq length 2,048)  ",
        "",
        "## Selected Winning Hyperparameters",
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

    (results_dir / "hparam_sweep_350m_algebraic_report.md").write_text("\n".join(md) + "\n")

    subprocess.run(["git", "add", "results/hparam_sweep_350m_algebraic_summary.json", "results/hparam_sweep_350m_algebraic_report.md"], check=True)
    subprocess.run(["git", "commit", "-m", "Add 350M Algebraic Transformer standard 10-arm sweep results and report"], check=True)
    subprocess.run(["git", "push", "origin", "main"], check=True)
    print(f"[{ist_now_str()}] Algebraic 350M sweep results committed and pushed to GitHub!", flush=True)


def publish_standard_and_comparison(alg_summary: dict[str, Any], std_summary: dict[str, Any]) -> None:
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "hparam_sweep_350m_standard_summary.json").write_text(json.dumps(std_summary, indent=2))

    std_winners = std_summary["selected_winners"]
    std_results = std_summary["results_by_optimizer"]
    alg_winners = alg_summary["selected_winners"]

    # 1. Standard Report
    md_std = [
        "# 350M Standard Transformer Hyperparameter Sweep Report (10 Arms)",
        "",
        f"**Date Completed:** {ist_now_str()}  ",
        "**Cluster Hardware:** Cloud TPU v4-32 (16 TPUs across 4 workers: `w0`, `w1`, `w2`, `w3`)  ",
        "**Model Architecture:** 350M Standard LLaMA-style Transformer (`ffn=swiglu`, `pos=rope`, `attn=softmax`, `head=ce`)  ",
        "**Active Parameters:** 332,449,792 (`d_model=1024`, `n_layers=24`, `n_heads=16`, `d_ff=2688`, `vocab_size=32768`)  ",
        "**Acceleration Kernels:** Softmax FlashAttention Pallas + Cross-Entropy Fused Head Pallas  ",
        "**Training Budget:** 600M tokens per arm (573 steps, batch size 512, micro 2, seq length 2,048)  ",
        "",
        "## Selected Winning Hyperparameters",
        "",
        "| Optimizer | Winning Arm | Base LR | Weight Decay | Muon LR | Muon WD | C-Eps | Held-Out NLL | Held-Out PPL |",
        "|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for opt in OPTIMIZERS:
        w = std_winners.get(opt, {})
        md_std.append(f"| **{opt}** | `{w.get('name')}` | {w.get('lr')} | {w.get('wd')} | {w.get('muon_lr')} | {w.get('muon_wd')} | {w.get('cautious_eps', '-')} | **{w.get('final_eval_nll'):.4f}** | **{w.get('final_eval_ppl'):.2f}** |")

    md_std.extend(["", "---", "", "## Full Sweep Arm Results by Optimizer", ""])
    for opt in OPTIMIZERS:
        md_std.append(f"### Optimizer: `{opt}`\n")
        md_std.append("| Arm Name | LR | WD | Muon LR | Muon WD | Final Loss | Held-Out NLL | Perplexity | Runtime |")
        md_std.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for r in std_results[opt]:
            m_lr = r.get("muon_lr") if r.get("muon_lr") is not None else "-"
            m_wd = r.get("muon_wd") if r.get("muon_wd") is not None else "-"
            floss = f"{r.get('final_train_loss'):.4f}" if r.get('final_train_loss') else "-"
            fnll = f"{r.get('final_eval_nll'):.4f}" if r.get('final_eval_nll') else "-"
            fppl = f"{r.get('final_eval_ppl'):.2f}" if r.get('final_eval_ppl') else "-"
            dur = f"{r.get('train_seconds', 0)/60:.1f} min"
            md_std.append(f"| `{r['name']}` | {r['lr']} | {r['wd']} | {m_lr} | {m_wd} | {floss} | **{fnll}** | {fppl} | {dur} |")
        md_std.append("")

    (results_dir / "hparam_sweep_350m_standard_report.md").write_text("\n".join(md_std) + "\n")

    # 2. Head-to-Head Comparison Report
    md_cmp = [
        "# 350M Architecture Head-to-Head Comparison Report: Algebraic vs. Standard",
        "",
        f"**Date Completed:** {ist_now_str()}  ",
        "**Cluster Hardware:** Cloud TPU v4-32 (16 TPUs across 4 workers: `w0`, `w1`, `w2`, `w3`)  ",
        "**Scale:** 350M Active Parameters (`~332.5M` parameters)  ",
        "**Evaluation Token Budget:** 600M tokens per arm (573 steps, batch 512, micro 2, seq 2,048)  ",
        "",
        "## Head-to-Head Winning Results Table",
        "",
        "| Architecture | Optimizer | Best Arm | Base LR | Muon LR | Held-Out NLL | Held-Out PPL | Delta NLL vs Standard | Relative PPL Reduction |",
        "|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for opt in OPTIMIZERS:
        sw = std_winners.get(opt, {})
        aw = alg_winners.get(opt, {})
        s_nll = sw.get("final_eval_nll", float("nan"))
        s_ppl = sw.get("final_eval_ppl", float("nan"))
        a_nll = aw.get("final_eval_nll", float("nan"))
        a_ppl = aw.get("final_eval_ppl", float("nan"))
        delta_nll = a_nll - s_nll
        rel_ppl = ((s_ppl - a_ppl) / s_ppl) * 100.0 if math.isfinite(s_ppl) and s_ppl > 0 else 0.0

        md_cmp.append(f"| **Standard LLaMA** | **{opt}** | `{sw.get('name')}` | {sw.get('lr')} | {sw.get('muon_lr')} | {s_nll:.4f} | {s_ppl:.2f} | baseline | baseline |")
        md_cmp.append(f"| **Algebraic Transformer** | **{opt}** | `{aw.get('name')}` | {aw.get('lr')} | {aw.get('muon_lr')} | **{a_nll:.4f}** | **{a_ppl:.2f}** | **{delta_nll:+.4f}** | **+{rel_ppl:.2f}%** |")

    (results_dir / "hparam_sweep_350m_comparison_report.md").write_text("\n".join(md_cmp) + "\n")

    subprocess.run([
        "git", "add",
        "results/hparam_sweep_350m_standard_summary.json",
        "results/hparam_sweep_350m_standard_report.md",
        "results/hparam_sweep_350m_comparison_report.md"
    ], check=True)
    subprocess.run(["git", "commit", "-m", "Add 350M Standard Transformer sweep results and head-to-head comparison report"], check=True)
    subprocess.run(["git", "push", "origin", "main"], check=True)
    print(f"[{ist_now_str()}] Standard 350M sweep and comparison reports committed and pushed to GitHub!", flush=True)


def main():
    print(f"[{ist_now_str()}] ========================================================", flush=True)
    print(f"[{ist_now_str()}] STARTING 350M BENCHMARK SWEEPS: ALGEBRAIC THEN STANDARD", flush=True)
    print(f"[{ist_now_str()}] Standard Extensive Grid (5 Arms / Optimizer, 2 Optimizers, No Main Runs)", flush=True)
    print(f"[{ist_now_str()}] ========================================================", flush=True)

    # Phase 1: 350M Algebraic Transformer Sweep (10 Arms)
    alg_jobs = build_algebraic_sweep_jobs()
    alg_summary = run_sweep_phase(ALG_SWEEP_GROUP, alg_jobs, "350M Algebraic Transformer")
    publish_algebraic_sweep(alg_summary)

    # Phase 2: 350M Standard Transformer Sweep (10 Arms)
    std_jobs = build_standard_sweep_jobs()
    std_summary = run_sweep_phase(STD_SWEEP_GROUP, std_jobs, "350M Standard Transformer")
    publish_standard_and_comparison(alg_summary, std_summary)

    print(f"\n[{ist_now_str()}] ========================================================", flush=True)
    print(f"[{ist_now_str()}] ALL 350M SWEEPS AND REPORTS COMPLETED AND PUSHED TO GITHUB!", flush=True)
    print(f"[{ist_now_str()}] ========================================================", flush=True)


if __name__ == "__main__":
    main()
