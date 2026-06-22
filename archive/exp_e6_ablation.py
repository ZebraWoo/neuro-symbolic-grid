#!/usr/bin/env python3
"""
E6: Ablation Study — Run all 5 ablation variants and collect results.

Usage:
  python experiments/exp_e6_ablation.py --epochs 20 --zones-per-split 3
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ABLATION_VARIANTS = {
    "ours_full":       [],                                    # Full model
    "ours_no_spike":   ["--no-spike"],                        # A1: No spike encoding
    "ours_no_sym":     ["--no-symbolic"],                     # A2: No symbolic rules
    "ours_no_phys":    ["--no-physics"],                      # A3: No physics constraints
    "ours_no_cl":      ["--no-closed-loop"],                  # A4: No closed-loop
    "ours_single_comp":["--no-multi-comp"],                   # A5: Single-compartment SNN
}


def run_ablation(tag, extra_args, base_args):
    """Run one ablation variant and extract best F1 from history."""
    cmd = [
        sys.executable, str(_PROJECT_ROOT / "experiments/exp_ours_full.py"),
        "--tag", tag,
    ] + base_args + extra_args

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_PROJECT_ROOT))

    if result.returncode != 0:
        logger.error("Ablation %s FAILED:\n%s\n%s", tag, result.stdout[-500:], result.stderr[-500:])
        return None

    # Parse best F1 from history file
    history_path = _PROJECT_ROOT / "outputs" / f"history_{tag}.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        best_f1 = max(history.get("val_macro_f1", [0])) if history.get("val_macro_f1") else 0
        return {"tag": tag, "best_macro_f1": best_f1, "history": history}
    return None


def main():
    parser = argparse.ArgumentParser(description="E6: Ablation Study")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--zones-per-split", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    base_args = [
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--zones-per-split", str(args.zones_per_split),
        "--device", args.device,
    ]

    results = {}
    for tag, extra in ABLATION_VARIANTS.items():
        logger.info("\n%s", "=" * 60)
        logger.info("Ablation: %s", tag)
        logger.info("%s", "=" * 60)
        r = run_ablation(tag, extra, base_args)
        if r:
            results[tag] = r
            logger.info("%s: Best Macro F1 = %.4f", tag, r["best_macro_f1"])
        else:
            logger.error("Skipping %s due to failure", tag)

    # Summary
    logger.info("\n%s", "=" * 60)
    logger.info("E6 Ablation Summary")
    logger.info("%s", "=" * 60)
    summary = {}
    for tag, r in results.items():
        logger.info("  %-25s: Macro F1 = %.4f", tag, r["best_macro_f1"])
        summary[tag] = {"best_macro_f1": r["best_macro_f1"]}

    out = Path(args.output_dir) / "e6_ablation_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved: %s", out)


if __name__ == "__main__":
    main()
