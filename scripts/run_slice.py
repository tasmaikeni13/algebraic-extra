#!/usr/bin/env python3
"""Run resumable data-parallel jobs across a TPU slice from worker zero.

The repository and Python environment must exist at the same paths on every
worker. Dataset arrays also need identical paths. --skip-done validates
completed outputs against --config before skipping a job.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

IST = timezone(timedelta(hours=5, minutes=30))


def ist_now_str() -> str:
    """Return formatted current timestamp in Indian Standard Time (IST)."""
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def validate_completed_job(run, job):
    """Require the requested recipe, full step budget, and finite measurements."""
    from dataclasses import asdict
    from algebraic.config import build
    from algebraic.train import get_args, parse_overrides
    from algebraic.benchmarking import load_jsonl
    args = get_args([*job["args"], "--distributed", "--out", str(run)])
    expected_args = vars(args)
    required = ["meta.json", "result.json", "log.jsonl"] + (["params.npz"] if args.save_params else [])
    if any(not (run / name).is_file() for name in required):
        raise ValueError(f"incomplete run: {run}")
    meta = json.loads((run / "meta.json").read_text())
    result = json.loads((run / "result.json").read_text())
    ignore = {"out", "resume", "log_every", "ckpt_every", "keep_ckpts"}
    if any(meta["args"].get(k) != v for k, v in expected_args.items() if k not in ignore):
        raise ValueError(f"completed run used a different recipe: {run}")
    config = asdict(build(args.preset, args.size, **{**parse_overrides(args.set), "seq_len": args.seq_len}))
    steps = args.steps or math.ceil(args.tokens / (args.batch * args.seq_len))
    losses = [row for row in load_jsonl(run / "log.jsonl") if "loss" in row]
    if (meta["config"] != config or result["config"] != config or not losses
            or losses[-1]["step"] != steps or not math.isfinite(losses[-1]["loss"])
            or meta["total_steps"] != steps or result.get("completed_steps") != steps
            or not math.isfinite(result["final_eval_nll"])):
        raise ValueError(f"incomplete or incompatible training result: {run}")
    return result


def remote_command(job, group, python=sys.executable):
    if not group or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in group):
        raise ValueError("group must be a simple directory name")
    name = job["name"]
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.+-" for c in name):
        raise ValueError("job name must be a simple directory name")
    out = ROOT / "runs" / group / name
    command = [python, "-m", "algebraic.train", "--distributed", *job["args"], "--out", str(out)]
    return (f"mkdir -p {shlex.quote(str(out.parent))} && cd {shlex.quote(str(ROOT))} && "
            f"JAX_PLATFORMS=tpu JAX_ENABLE_X64=0 PYTHONPATH={shlex.quote(str(ROOT / 'src'))} "
            f"{shlex.join(command)} > {shlex.quote(str(out))}.\"$(hostname)\".log 2>&1")


def print_worker_failure_logs(hosts: list[str], log_prefix: Path, lines: int = 35) -> None:
    """Print the tail of worker logs across hosts upon job failure."""
    print(f"\n{'='*70}\n[ERROR DIAGNOSTICS] Tail of worker logs for {log_prefix.name}:\n{'='*70}", file=sys.stderr)
    for h in hosts:
        cmd = f"tail -n {lines} {shlex.quote(str(log_prefix))}.*.log 2>/dev/null || true"
        try:
            out = subprocess.check_output(["ssh", h, cmd], text=True, timeout=10).strip()
            if out:
                print(f"[{h}]:\n{out}\n{'-'*50}", file=sys.stderr)
        except Exception as e:
            print(f"[{h}]: (failed to fetch logs: {e})", file=sys.stderr)


def prune_storage_after_arm(hosts: list[str], arm_dir: Path) -> None:
    """Ensure strict storage consciousness: remove checkpoints/weights and transient logs."""
    # Local worker 0 cleanup
    for p in arm_dir.glob("*.npz"):
        p.unlink(missing_ok=True)
    shutil.rmtree(arm_dir / "ckpt", ignore_errors=True)

    # Remote workers cleanup
    for h in hosts:
        cmds = [
            f"rm -rf {shlex.quote(str(arm_dir))}/ckpt {shlex.quote(str(arm_dir))}/*.npz /tmp/tpu_logs/* 2>/dev/null || true"
        ]
        if h != hosts[0]:
            cmds.append(f"rm -f {shlex.quote(str(arm_dir))}.*.log 2>/dev/null || true")
        cmd_str = " ; ".join(cmds)
        subprocess.run(["ssh", h, cmd_str], timeout=20, check=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jobs", type=Path)
    p.add_argument("--group", required=True)
    p.add_argument("--hosts", default="w0,w1,w2,w3")
    p.add_argument("--no-sync", action="store_true")
    p.add_argument("--skip-done", action="store_true")
    a = p.parse_args()
    hosts = a.hosts.split(",")
    jobs = json.loads(a.jobs.read_text())
    if not hosts or len(hosts) != len(set(hosts)) or any(not h or h.startswith('-') for h in hosts):
        p.error("hosts must be distinct SSH host names")
    names = [j["name"] for j in jobs]
    if len(names) != len(set(names)):
        p.error("job names must be unique")
    commands = [remote_command(job, a.group) for job in jobs]
    if not a.no_sync:
        subprocess.run(["bash", str(ROOT / "scripts/setup_workers.sh"), *hosts], cwd=ROOT, check=True)
    root = ROOT / "runs" / a.group
    root.mkdir(parents=True, exist_ok=True)

    total_jobs = len(jobs)
    completed_runs: list[dict] = []
    status_file = root / "status.json"
    sweep_start_time = time.monotonic()

    def update_status(extra: dict | None = None):
        st = {
            "updated_at_ist": ist_now_str(),
            "group": a.group,
            "total_arms": total_jobs,
            "completed_count": len(completed_runs),
            "completed_arms": completed_runs,
        }
        if extra:
            st.update(extra)
        try:
            status_file.write_text(json.dumps(st, indent=2))
        except Exception:
            pass

    update_status({"status": "starting", "current_arm": None})

    for job_idx, (job, command) in enumerate(zip(jobs, commands)):
        name = job["name"]
        arm_dir = root / name
        if a.skip_done and (arm_dir / "result.json").exists():
            try:
                res = validate_completed_job(arm_dir, job)
                print(f"[{ist_now_str()}] Reuse complete training run {name} (NLL: {res.get('final_eval_nll', 'N/A')})", flush=True)
                completed_runs.append({
                    "name": name,
                    "final_eval_nll": res.get("final_eval_nll"),
                    "final_train_loss": res.get("final_train_loss"),
                    "status": "reused",
                })
                update_status({"current_arm": name, "status": "reused"})
                continue
            except Exception as e:
                print(f"Validation failed for {name} ({e}), rerunning...", flush=True)

        print(f"\n[{ist_now_str()}] >>> Starting Arm [{job_idx + 1}/{total_jobs}]: {name}", flush=True)
        t0 = time.monotonic()
        processes = [subprocess.Popen(["ssh", h, command]) for h in hosts]
        log_jsonl = arm_dir / "log.jsonl"
        last_reported_step = -1
        total_steps = 573  # default 600M tokens budget at 512 batch x 2048 seq_len

        # Extract total_steps from job args if present
        if "--steps" in job["args"]:
            try:
                total_steps = int(job["args"][job["args"].index("--steps") + 1])
            except Exception:
                pass

        try:
            while any(pr.poll() is None for pr in processes):
                failed = [pr.returncode for pr in processes if pr.poll() not in (None, 0)]
                if failed:
                    print_worker_failure_logs(hosts, root / name)
                    raise RuntimeError(f"{name}: worker failed with exit codes {failed}; see logs above")

                # Read live telemetry from log.jsonl
                if log_jsonl.is_file():
                    try:
                        lines = [line.strip() for line in log_jsonl.read_text().strip().split("\n") if line.strip()]
                        loss_rows = []
                        for l in lines:
                            try:
                                d = json.loads(l)
                                if "loss" in d and "step" in d:
                                    loss_rows.append(d)
                            except Exception:
                                pass
                        if loss_rows:
                            latest = loss_rows[-1]
                            cur_step = latest["step"]
                            if cur_step != last_reported_step:
                                last_reported_step = cur_step
                                elapsed = time.monotonic() - t0
                                rate = latest.get("rate", 0.0)
                                loss_val = latest["loss"]
                                gnorm_val = latest.get("gnorm", 0.0)
                                lr_val = latest.get("lr", 0.0)
                                pct = (cur_step / total_steps) * 100.0

                                # Calculate ETAs
                                step_rate = cur_step / max(elapsed, 1e-6)
                                remaining_steps = max(0, total_steps - cur_step)
                                arm_rem_sec = remaining_steps / max(step_rate, 1e-6)
                                arm_eta_dt = datetime.now(IST) + timedelta(seconds=arm_rem_sec)

                                # Total sweep remaining
                                avg_arm_sec = (time.monotonic() - sweep_start_time) / max(len(completed_runs) + (cur_step / total_steps), 1e-6)
                                remaining_arms = (total_jobs - job_idx - 1)
                                sweep_rem_sec = arm_rem_sec + remaining_arms * avg_arm_sec
                                sweep_eta_dt = datetime.now(IST) + timedelta(seconds=sweep_rem_sec)

                                print(
                                    f"[{ist_now_str()}] [{name} | Arm {job_idx+1}/{total_jobs}] "
                                    f"step {cur_step}/{total_steps} ({pct:4.1f}%) | "
                                    f"loss: {loss_val:.4f} | gnorm: {gnorm_val:.3f} | lr: {lr_val:.2e} | "
                                    f"{rate/1e3:4.0f}K tok/s | "
                                    f"Arm ETA: {arm_eta_dt.strftime('%H:%M:%S IST')} | "
                                    f"Sweep ETA: {sweep_eta_dt.strftime('%H:%M:%S IST')}",
                                    flush=True
                                )

                                update_status({
                                    "status": "training",
                                    "current_arm": name,
                                    "arm_index": job_idx + 1,
                                    "step": cur_step,
                                    "total_steps": total_steps,
                                    "progress_pct": round(pct, 1),
                                    "latest_loss": loss_val,
                                    "latest_gnorm": gnorm_val,
                                    "latest_lr": lr_val,
                                    "tok_per_sec": rate,
                                    "arm_eta_ist": arm_eta_dt.strftime("%Y-%m-%d %H:%M:%S IST"),
                                    "sweep_eta_ist": sweep_eta_dt.strftime("%Y-%m-%d %H:%M:%S IST"),
                                })
                    except Exception:
                        pass
                time.sleep(2)

            codes = [pr.wait() for pr in processes]
            if any(codes):
                print_worker_failure_logs(hosts, root / name)
                raise RuntimeError(f"{name}: worker exit codes {codes}")

        finally:
            for pr in processes:
                if pr.poll() is None:
                    pr.terminate()
            for pr in processes:
                pr.wait()

        elapsed_min = (time.monotonic() - t0) / 60.0
        print(f"[{ist_now_str()}] [Arm {job_idx+1}/{total_jobs} Completed] {name} in {elapsed_min:.1f} min", flush=True)

        # Read result metrics
        res_info = {}
        if (arm_dir / "result.json").is_file():
            try:
                res_data = json.loads((arm_dir / "result.json").read_text())
                res_info = {
                    "final_eval_nll": res_data.get("final_eval_nll"),
                    "final_eval_ppl": res_data.get("final_eval_ppl"),
                    "final_train_loss": res_data.get("final_train_loss"),
                }
                print(f"  --> Final Held-Out NLL: {res_data.get('final_eval_nll'):.4f} (PPL: {res_data.get('final_eval_ppl'):.2f})", flush=True)
            except Exception:
                pass

        completed_runs.append({
            "name": name,
            "duration_min": round(elapsed_min, 2),
            **res_info,
            "completed_at_ist": ist_now_str(),
        })

        # Storage conscious: delete checkpoints and weights across all cluster nodes
        prune_storage_after_arm(hosts, arm_dir)
        update_status({"status": "arm_done", "current_arm": name})

    print(f"\n[{ist_now_str()}] All {total_jobs} sweep arms completed successfully!", flush=True)
    update_status({"status": "all_completed", "completed_at_ist": ist_now_str()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
