#!/usr/bin/env python3
"""Background monitor for TPU hyperparameter sweeps.

Continuously tracks progress across cluster workers, mirrors outputs to worker zero,
calculates real-time ETA in Indian Standard Time (IST), updates sweep_status.json,
and enforces storage consciousness by cleaning checkpoints across workers.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
IST = timezone(timedelta(hours=5, minutes=30))
HOSTS = ("w0", "w1", "w2", "w3")
SWEEP_GROUP = "hparam_sweep_125m"


def ist_now() -> datetime:
    return datetime.now(IST)


def ist_now_str() -> str:
    return ist_now().strftime("%Y-%m-%d %H:%M:%S IST")


def find_coordinator_host(hosts: tuple[str, ...], group: str, arm_name: str) -> str:
    """Find which worker is currently acting as JAX process index 0 (writing log.jsonl)."""
    # Check remote workers first since w0 is the coordinator runner host
    for h in ("w2", "w1", "w3", "w0"):
        check = subprocess.run(
            ["ssh", h, f"test -f /home/tasma/algebraic-extra/runs/{group}/{arm_name}/log.jsonl"],
            check=False, timeout=5
        )
        if check.returncode == 0:
            return h
    return "w2"


def main():
    sweep_dir = ROOT / "runs" / SWEEP_GROUP
    sweep_dir.mkdir(parents=True, exist_ok=True)
    jobs_file = sweep_dir / "jobs.json"
    status_file = sweep_dir / "sweep_status.json"

    if not jobs_file.is_file():
        time.sleep(2)

    jobs = json.loads(jobs_file.read_text()) if jobs_file.is_file() else []
    total_arms = len(jobs) if jobs else 15
    arm_names = [j["name"] for j in jobs] if jobs else []

    sweep_start_time = time.monotonic()
    completed_arms: list[dict] = []
    last_logged_step = -1

    print(f"[{ist_now_str()}] Sweep monitor daemon started. Tracking {total_arms} arms.", flush=True)

    while True:
        try:
            # 1. Identify which arm is currently training
            current_arm = None
            current_arm_idx = -1

            # Check if any training process is alive
            ps_out = subprocess.check_output(
                ["ssh", "w0", "ps aux | grep algebraic.train | grep -v grep || true"],
                text=True, timeout=10
            )

            if "algebraic.train" in ps_out:
                for idx, name in enumerate(arm_names):
                    if (sweep_dir / name / "result.json").is_file():
                        continue
                    # Check if arm directory or log exists on any worker
                    current_arm = name
                    current_arm_idx = idx
                    break

            if not current_arm:
                # Check completed arms
                for idx, name in enumerate(arm_names):
                    arm_local = sweep_dir / name
                    res_file = arm_local / "result.json"
                    if res_file.is_file() and not any(c["name"] == name for c in completed_arms):
                        try:
                            res = json.loads(res_file.read_text())
                            completed_arms.append({
                                "name": name,
                                "final_eval_nll": res.get("final_eval_nll"),
                                "final_eval_ppl": res.get("final_eval_ppl"),
                                "final_train_loss": res.get("final_train_loss"),
                                "train_seconds": res.get("train_seconds"),
                            })
                        except Exception:
                            pass

                if len(completed_arms) >= total_arms and total_arms > 0:
                    status_data = {
                        "status": "ALL_COMPLETED",
                        "updated_at_ist": ist_now_str(),
                        "completed_arms_count": len(completed_arms),
                        "total_arms": total_arms,
                        "completed_arms": completed_arms,
                    }
                    status_file.write_text(json.dumps(status_data, indent=2))
                    print(f"[{ist_now_str()}] All {total_arms} arms completed! Daemon exiting.", flush=True)
                    break

                time.sleep(5)
                continue

            # 2. Sync artifacts from coordinator host to w0
            coord = find_coordinator_host(HOSTS, SWEEP_GROUP, current_arm)
            remote_arm_dir = f"/home/tasma/algebraic-extra/runs/{SWEEP_GROUP}/{current_arm}"
            local_arm_dir = sweep_dir / current_arm
            local_arm_dir.mkdir(parents=True, exist_ok=True)

            if coord != "w0":
                subprocess.run(
                    ["rsync", "-q", "-a", f"{coord}:{remote_arm_dir}/", f"{local_arm_dir}/"],
                    timeout=15, check=False
                )

            # 3. Read log.jsonl on w0
            log_jsonl = local_arm_dir / "log.jsonl"
            latest_step = 0
            latest_loss = None
            latest_gnorm = None
            latest_lr = None
            latest_rate = None
            total_steps = 573

            if log_jsonl.is_file():
                try:
                    lines = [l.strip() for l in log_jsonl.read_text().split("\n") if l.strip()]
                    loss_records = []
                    for l in lines:
                        try:
                            d = json.loads(l)
                            if "loss" in d and "step" in d:
                                loss_records.append(d)
                        except Exception:
                            pass
                    if loss_records:
                        latest = loss_records[-1]
                        latest_step = latest["step"]
                        latest_loss = latest["loss"]
                        latest_gnorm = latest.get("gnorm")
                        latest_lr = latest.get("lr")
                        latest_rate = latest.get("tok_per_s")
                except Exception:
                    pass

            # 4. Check if arm just completed
            result_file = local_arm_dir / "result.json"
            if result_file.is_file():
                try:
                    res = json.loads(result_file.read_text())
                    if not any(c["name"] == current_arm for c in completed_arms):
                        completed_arms.append({
                            "name": current_arm,
                            "final_eval_nll": res.get("final_eval_nll"),
                            "final_eval_ppl": res.get("final_eval_ppl"),
                            "final_train_loss": res.get("final_train_loss"),
                            "train_seconds": res.get("train_seconds"),
                        })
                        # Storage consciousness: prune checkpoints & weights across all workers
                        for h in HOSTS:
                            subprocess.run([
                                "ssh", h,
                                f"rm -rf {remote_arm_dir}/ckpt {remote_arm_dir}/*.npz /tmp/tpu_logs/* 2>/dev/null || true"
                            ], timeout=15, check=False)
                        print(f"[{ist_now_str()}] [Arm {current_arm_idx+1}/{total_arms}] {current_arm} finished! NLL: {res.get('final_eval_nll'):.4f}", flush=True)
                except Exception:
                    pass

            # 5. Compute real-time IST ETA
            step_tokens = 512 * 2048  # 1,048,576
            effective_rate = latest_rate if (latest_rate and latest_rate > 100000) else 1575000.0
            step_duration = step_tokens / effective_rate  # ~0.665s per step
            eval_overhead = 4.0 * (100 / total_steps)  # periodic eval overhead

            remaining_step_in_arm = max(0, total_steps - latest_step)
            arm_seconds_left = remaining_step_in_arm * (step_duration + 0.05)
            arm_eta_dt = ist_now() + timedelta(seconds=arm_seconds_left)

            est_arm_total_sec = total_steps * (step_duration + 0.05) + 30  # ~410s per arm including compile
            remaining_arms = max(0, total_arms - (current_arm_idx + 1))
            sweep_seconds_left = arm_seconds_left + remaining_arms * est_arm_total_sec
            sweep_eta_dt = ist_now() + timedelta(seconds=sweep_seconds_left)

            progress_pct = (latest_step / total_steps) * 100.0 if total_steps else 0.0

            # 6. Update sweep_status.json
            status_data = {
                "status": "RUNNING",
                "updated_at_ist": ist_now_str(),
                "current_arm": current_arm,
                "current_arm_index": current_arm_idx + 1,
                "total_arms": total_arms,
                "current_step": latest_step,
                "total_steps": total_steps,
                "arm_progress_pct": round(progress_pct, 1),
                "latest_loss": latest_loss,
                "latest_gnorm": latest_gnorm,
                "latest_lr": latest_lr,
                "tok_per_sec": latest_rate,
                "arm_eta_ist": arm_eta_dt.strftime("%Y-%m-%d %H:%M:%S IST"),
                "total_sweep_eta_ist": sweep_eta_dt.strftime("%Y-%m-%d %H:%M:%S IST"),
                "completed_arms_count": len(completed_arms),
                "completed_arms": completed_arms,
            }
            status_file.write_text(json.dumps(status_data, indent=2))

            if latest_step != last_logged_step and latest_loss is not None:
                last_logged_step = latest_step
                print(
                    f"[{ist_now_str()}] [Arm {current_arm_idx+1}/{total_arms}: {current_arm}] "
                    f"step {latest_step}/{total_steps} ({progress_pct:4.1f}%) | "
                    f"loss: {latest_loss:.4f} | gnorm: {latest_gnorm:.3f} | "
                    f"Arm ETA: {arm_eta_dt.strftime('%H:%M:%S IST')} | "
                    f"Sweep ETA: {sweep_eta_dt.strftime('%H:%M:%S IST')}",
                    flush=True
                )

        except Exception as e:
            pass

        time.sleep(5)


if __name__ == "__main__":
    main()
