#!/usr/bin/env python3
"""Stage completed comparison checkpoints and results for an explicit Hub upload.

Set --runs-root and --results-root for the desired model scale. This utility
is optional; running training or evaluation never uploads or publishes data.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def generate_model_card(out_path, comparison_md=None):
    header = """---
license: apache-2.0
tags:
- language-modeling
- algebraic-transformer
- jax
- pallas
- tpu
---

# Algebraic Transformer checkpoints

Matched standard and pure algebraic Transformer language models trained on FineWeb-Edu. Training metadata in each checkpoint directory records the model dimensions, token budget, and seed.

The algebraic model uses Cayley rotations, ALU-GLU, learned-sink algebraic attention, and a normalized algebraic output distribution. Its model-objective gradient uses arithmetic and reciprocal roots; initialization, the optimizer schedule, and metric reporting are outside this claim.

Evaluation includes ARC-Easy, ARC-Challenge, HellaSwag, PIQA, OpenBookQA, WinoGrande, BoolQ, LAMBADA, and four bounded-context recall tasks. Consult the tables and checkpoint metadata for measured model scales and seeds. Smoke verification is not a quality benchmark.

"""
    out_path.write_text(header + (comparison_md.read_text() if comparison_md and comparison_md.is_file() else ""))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default="tasmaikeni13/algebraic-transformer")
    p.add_argument("--runs-root", type=Path, default=ROOT / "runs/main")
    p.add_argument("--results-root", type=Path, default=ROOT / "results/main")
    p.add_argument("--private", action="store_true")
    p.parse_args(argv)
    print("INFO: Hugging Face upload is disabled in accordance with R5 safety constraints (no external uploads).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
