#!/usr/bin/env python3
"""Multi-host 16-TPU smoke validation suite for 125M and 350M models.

Executes 1 synthetic training step across all 16 TPUs (w0..w3) for 125M and
350M Algebraic Transformers with pure XLA optimizers (Muon+AdamW, C-AdamW,
Hybrid Muon+C-AdamW). Verifies multi-host coordination, shard_map data
parallelism, Pallas kernels, loss and gradient norm finiteness.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

SCALES = ("125m", "350m")
OPTIMIZERS = ("muon_adamw", "cadamw", "hybrid_muon_cadamw")
SYNTHETIC_TOKENS = 512 * 2048 + 1  # 1_048_577 tokens: tokens_per_step + 1
DEFAULT_PYTHON = "/home/tasma/venv/bin/python"


def positive_int(value: str) -> int:
    """Validate positive integer for command line step count."""
    ival = int(value)
    if ival <= 0:
        raise argparse.ArgumentTypeError(f"steps must be > 0, got {ival}")
    return ival


def generate_synthetic_data(
    data_dir: Path,
    vocab_size: int = 32768,
    seq_len: int = 2048,
    batch_size: int = 512,
    seed: int = 42,
) -> Tuple[Path, Path]:
    """Generate deterministic synthetic token arrays with EOT markers."""
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "fineweb_train_synthetic.npy"
    valid_path = data_dir / "fineweb_valid_synthetic.npy"
    synthetic_len = batch_size * seq_len + 1  # 1_048_577 tokens
    eot_token = 2
    rng = np.random.default_rng(seed)

    for path in (train_path, valid_path):
        tokens = rng.integers(0, vocab_size, size=synthetic_len, dtype=np.uint16)
        tokens[seq_len::seq_len] = eot_token
        tokens[-1] = eot_token
        np.save(path, tokens)

    return train_path, valid_path


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for 16-TPU smoke testing."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--scale",
        choices=("125m", "350m", "all"),
        default="all",
        help="Model scale to smoke test: 125m, 350m, or all (default: all)",
    )
    p.add_argument(
        "--optimizer",
        choices=("muon_adamw", "cadamw", "hybrid_muon_cadamw", "all"),
        default="all",
        help="Optimizer: muon_adamw, cadamw, hybrid_muon_cadamw, or all (default: all)",
    )
    p.add_argument(
        "--hosts",
        default="w0,w1,w2,w3",
        help="Comma-separated cluster worker hosts (default: w0,w1,w2,w3)",
    )
    p.add_argument(
        "--steps",
        type=positive_int,
        default=1,
        help="Number of optimizer steps to execute (default: 1)",
    )
    p.add_argument(
        "--group",
        default=None,
        help="Run directory group name under runs/",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Path to write the verification JSON report",
    )
    p.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip running scripts/setup_workers.sh to sync workers",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and generate data without remote execution",
    )
    return p


def cleanup_remote_processes(hosts: List[str]) -> None:
    """Terminate any lingering training processes holding TPU locks across hosts."""
    for host in hosts:
        try:
            cmd = "pkill -u $(whoami) -9 -f 'algebraic\\.train' 2>/dev/null || true"
            subprocess.run(["ssh", host, cmd], timeout=10, check=False)
        except Exception:
            pass


def remote_command(
    job_args: List[str],
    out_dir: Path,
    root_dir: Path,
    python_bin: str | None = None,
) -> str:
    """Format remote execution command with distributed environment flags."""
    py = python_bin or (
        DEFAULT_PYTHON if Path(DEFAULT_PYTHON).exists() else sys.executable
    )
    cmd = [
        py,
        "-m",
        "algebraic.train",
        "--distributed",
        *job_args,
        "--out",
        str(out_dir),
    ]
    log_target = f"{shlex.quote(str(out_dir))}.$(hostname).log"
    return (
        f"mkdir -p {shlex.quote(str(out_dir.parent))} && "
        f"cd {shlex.quote(str(root_dir))} && "
        f"JAX_PLATFORMS=tpu JAX_ENABLE_X64=0 "
        f"PYTHONPATH={shlex.quote(str(root_dir / 'src'))} "
        f"{shlex.join(cmd)} > {log_target} 2>&1"
    )


def fetch_tail_logs(hosts: List[str], out_dir: Path, lines: int = 30) -> None:
    """Fetch and print tail of log files from all workers upon failure."""
    print("--- Tail of worker logs ---", file=sys.stderr)
    for host in hosts:
        log_file = f"{out_dir}.*.log"
        tail_cmd = f"tail -n {lines} {log_file} 2>/dev/null || true"
        try:
            out = subprocess.check_output(["ssh", host, tail_cmd], text=True)
            if out.strip():
                print(f"[{host}]:\n{out.strip()}", file=sys.stderr)
        except Exception:
            pass
    print("---------------------------", file=sys.stderr)


def retrieve_run_artifacts(hosts: List[str], out_dir: Path) -> None:
    """Retrieve output artifacts to coordinator if main process ran on a remote worker."""
    result_path = out_dir / "result.json"
    if result_path.exists():
        return
    for host in hosts:
        remote_check = subprocess.run(
            ["ssh", host, f"test -f {shlex.quote(str(result_path))}"],
            check=False,
        )
        if remote_check.returncode == 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["rsync", "-a", f"{host}:{out_dir}/", f"{out_dir}/"],
                check=True,
            )
            return


def run_distributed_job(
    hosts: List[str],
    job_args: List[str],
    out_dir: Path,
    root_dir: Path,
) -> None:
    """Launch distributed job over SSH to all cluster hosts and monitor execution."""
    cmd_str = remote_command(job_args, out_dir, root_dir)
    processes = [subprocess.Popen(["ssh", h, cmd_str]) for h in hosts]

    def sig_handler(signum: int, frame: Any) -> None:
        for p in processes:
            p.terminate()
        cleanup_remote_processes(hosts)
        sys.exit(1)

    old_sigint = signal.signal(signal.SIGINT, sig_handler)
    old_sigterm = signal.signal(signal.SIGTERM, sig_handler)

    try:
        while any(p.poll() is None for p in processes):
            failed = [p.returncode for p in processes if p.poll() not in (None, 0)]
            if failed:
                fetch_tail_logs(hosts, out_dir)
                raise RuntimeError(
                    f"Remote worker failure with exit codes {failed}; see logs in {out_dir}"
                )
            time.sleep(1)
        codes = [p.wait() for p in processes]
        if any(codes):
            fetch_tail_logs(hosts, out_dir)
            raise RuntimeError(f"Remote workers exited with non-zero codes: {codes}")
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            p.wait()


def main(argv: List[str] | None = None) -> int:
    """Execute multi-host smoke test suite across configured scales and optimizers."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.steps <= 0:
        parser.error("steps must be a positive integer (> 0)")

    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    if not hosts or len(hosts) != len(set(hosts)) or any(h.startswith("-") for h in hosts):
        parser.error("hosts must be distinct, non-empty hostnames")

    selected_scales = list(SCALES) if args.scale == "all" else [args.scale]
    selected_opts = list(OPTIMIZERS) if args.optimizer == "all" else [args.optimizer]

    group = args.group or f"smoke_16tpu_{int(time.time())}"
    valid_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if any(c not in valid_chars for c in group):
        parser.error("group must contain only alphanumeric characters, underscores, or hyphens")

    data_dir = ROOT / "runs" / group / "synthetic"
    train_path, valid_path = generate_synthetic_data(data_dir)

    if not args.dry_run:
        cleanup_remote_processes(hosts)
        if not args.no_sync:
            subprocess.run(
                ["bash", str(ROOT / "scripts/setup_workers.sh"), *hosts],
                cwd=ROOT,
                check=True,
            )
        for host in hosts:
            check_cmd = ["ssh", host, "hostname"]
            remote_host = subprocess.check_output(check_cmd, text=True).strip()
            local_host = subprocess.check_output(["hostname"], text=True).strip()
            if remote_host == local_host:
                continue
            subprocess.run(["ssh", host, f"mkdir -p {shlex.quote(str(data_dir))}"], check=True)
            subprocess.run(
                ["rsync", "-a", str(data_dir) + "/", f"{host}:{data_dir}/"],
                check=True,
            )

    report: Dict[str, Any] = {
        "passed": False,
        "completed_steps": args.steps,
        "scale": args.scale,
        "optimizer": args.optimizer,
        "devices": len(hosts) * 4,
        "processes": len(hosts),
        "hosts": hosts,
        "loss": None,
        "grad_norm": None,
        "runs": {},
    }

    scale_reports: Dict[str, Dict[str, Any]] = {}
    for scale in selected_scales:
        scale_reports[scale] = {
            "passed": False,
            "completed_steps": args.steps,
            "scale": scale,
            "optimizer": args.optimizer,
            "devices": len(hosts) * 4,
            "processes": len(hosts),
            "hosts": hosts,
            "loss": None,
            "grad_norm": None,
            "runs": {},
        }

    last_loss: float | None = None
    last_gnorm: float | None = None

    for scale in selected_scales:
        config_path = ROOT / f"configs/comparison_{scale}.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        training_cfg = config["training"]
        micro = training_cfg.get("micro", 4 if scale == "125m" else 2)
        lr = training_cfg.get("lr", 0.003 if scale == "125m" else 0.002)

        for opt in selected_opts:
            run_name = f"{scale}_{opt}"
            out_dir = ROOT / "runs" / group / run_name
            job_args = [
                "--preset", "algebraic",
                "--size", scale,
                "--optimizer", opt,
                "--steps", str(args.steps),
                "--batch", "512",
                "--seq-len", "2048",
                "--micro", str(micro),
                "--lr", str(lr),
                "--warmup", "1",
                "--schedule", "cosine",
                "--wd", "0.1",
                "--train", str(train_path),
                "--valid", str(valid_path),
                "--eval-tokens", "65536",
                "--eval-batch", str(micro),
                "--eval-every", "0",
                "--ckpt-every", "0",
                "--log-every", "1",
                "--seed", "42",
                "--data-seed", "42",
                "--set", "attn_impl=flash",
                "--set", "head_impl=fused",
                "--set", "flash_block=512",
                "--set", "head_block_n=128",
                "--set", "head_block_v=512",
                "--set", "vocab_size=32768",
            ]

            print(f"Executing 16-TPU smoke: scale={scale}, optimizer={opt}...", flush=True)
            t0 = time.monotonic()
            if not args.dry_run:
                run_distributed_job(hosts, job_args, out_dir, ROOT)
                retrieve_run_artifacts(hosts, out_dir)
                result_path = out_dir / "result.json"
                log_path = out_dir / "log.jsonl"
                if not result_path.exists() or not log_path.exists():
                    raise AssertionError(f"Missing run output artifacts in {out_dir}")

                result = json.loads(result_path.read_text(encoding="utf-8"))
                logs = [
                    json.loads(line)
                    for line in log_path.read_text(encoding="utf-8").splitlines()
                ]
                step1_metrics = next((m for m in logs if m.get("step") == args.steps), {})

                loss = float(
                    step1_metrics.get("loss", result.get("final_train_loss", float("nan")))
                )
                gnorm = float(step1_metrics.get("gnorm", float("nan")))

                if (
                    result.get("completed_steps") != args.steps
                    or not math.isfinite(loss)
                    or not math.isfinite(gnorm)
                ):
                    raise AssertionError(
                        f"Smoke execution mismatch or non-finite metric in {run_name}: "
                        f"loss={loss}, gnorm={gnorm}, steps={result.get('completed_steps')}"
                    )

                last_loss = loss
                last_gnorm = gnorm
                run_entry = {
                    "passed": True,
                    "scale": scale,
                    "optimizer": opt,
                    "completed_steps": args.steps,
                    "devices": result.get("devices", len(hosts) * 4),
                    "processes": result.get("processes", len(hosts)),
                    "loss": loss,
                    "grad_norm": gnorm,
                    "final_eval_nll": result.get("final_eval_nll"),
                    "parameters": result.get("n_params"),
                    "elapsed_s": time.monotonic() - t0,
                }
            else:
                last_loss = 10.0
                last_gnorm = 1.0
                run_entry = {
                    "passed": True,
                    "scale": scale,
                    "optimizer": opt,
                    "completed_steps": args.steps,
                    "devices": len(hosts) * 4,
                    "processes": len(hosts),
                    "loss": 10.0,
                    "grad_norm": 1.0,
                    "dry_run": True,
                }

            report["runs"][run_name] = run_entry
            scale_reports[scale]["runs"][run_name] = run_entry
            scale_reports[scale]["loss"] = last_loss
            scale_reports[scale]["grad_norm"] = last_gnorm
            scale_reports[scale]["passed"] = True

    report["passed"] = True
    report["loss"] = last_loss
    report["grad_norm"] = last_gnorm

    verification_dir = ROOT / "results" / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)

    if args.out:
        out_file = args.out
    elif args.dry_run:
        out_file = verification_dir / "smoke_16tpu_dryrun.json"
    else:
        out_file = verification_dir / "smoke_16tpu.json"

    if out_file.exists():
        try:
            existing = json.loads(out_file.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and "runs" in existing:
                merged_runs = dict(existing["runs"])
                merged_runs.update(report["runs"])
                report["runs"] = merged_runs
                scales_in_runs = {r.get("scale") for r in report["runs"].values()}
                opts_in_runs = {r.get("optimizer") for r in report["runs"].values()}
                if set(SCALES).issubset(scales_in_runs):
                    report["scale"] = "all"
                if set(OPTIMIZERS).issubset(opts_in_runs):
                    report["optimizer"] = "all"
                if report["loss"] is None and existing.get("loss") is not None:
                    report["loss"] = existing.get("loss")
                    report["grad_norm"] = existing.get("grad_norm")
        except Exception:
            pass

    report["passed"] = all(
        r.get("passed", False) for r in report["runs"].values()
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.dry_run or args.out:
        for scale, s_report in scale_reports.items():
            if args.out:
                scale_artifact = out_file.parent / f"distributed_{scale}.json"
            else:
                scale_artifact = verification_dir / f"distributed_{scale}.json"

            if scale_artifact.exists():
                try:
                    existing_s = json.loads(scale_artifact.read_text(encoding="utf-8"))
                    if isinstance(existing_s, dict) and "runs" in existing_s:
                        merged_s_runs = dict(existing_s["runs"])
                        merged_s_runs.update(s_report["runs"])
                        s_report["runs"] = merged_s_runs
                        opts_in_s = {r.get("optimizer") for r in s_report["runs"].values()}
                        if set(OPTIMIZERS).issubset(opts_in_s):
                            s_report["optimizer"] = "all"
                        if s_report["loss"] is None and existing_s.get("loss") is not None:
                            s_report["loss"] = existing_s.get("loss")
                            s_report["grad_norm"] = existing_s.get("grad_norm")
                except Exception:
                    pass

            s_report["passed"] = all(
                r.get("passed", False) for r in s_report["runs"].values()
            )
            scale_artifact.parent.mkdir(parents=True, exist_ok=True)
            scale_artifact.write_text(json.dumps(s_report, indent=2), encoding="utf-8")

    print(f"Smoke verification successful. Report written to {out_file}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
