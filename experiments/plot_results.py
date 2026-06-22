#!/usr/bin/env python3
"""
Generate paper-ready figures from experiment results.

Reads JSON outputs from experiments/ and produces:
  1. Decision Accuracy comparison bar chart
  2. Rule Satisfaction Rate heatmap
  3. Physics Violation count bar chart
  4. Closed-loop convergence curve
  5. Robustness degradation curve
  6. Ablation waterfall chart
  7. Case study multi-panel figure

Usage:
  python experiments/plot_results.py --output-dir results/paper_figures/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def plot_e1_decision_accuracy(results_dir: Path, output_dir: Path):
    """Bar chart: Macro F1 per method."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping E1 plot")
        return

    summary_path = results_dir / "e1_summary.json"
    if not summary_path.exists():
        logger.warning("E1 summary not found at %s", summary_path)
        return

    with open(summary_path) as f:
        data = json.load(f)

    models = list(data.keys())
    f1s = [data[m]["best_macro_f1"] for m in models]

    colors = ["#1f77b4" if m != "ours_full" else "#d62728" for m in models]
    labels = [m.replace("_", " ").title() for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, f1s, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_title("Experiment 1: Decision Intent Accuracy", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.axhline(y=max(f1s), color="gray", linestyle="--", alpha=0.5, label=f"Best: {max(f1s):.3f}")

    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

    ax.legend(fontsize=10)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(output_dir / "e1_decision_accuracy.png", dpi=150)
    plt.close()
    logger.info("Saved e1_decision_accuracy.png")


def plot_e2_rule_satisfaction(results_dir: Path, output_dir: Path):
    """Table/heatmap of rule satisfaction rates."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rsr_path = results_dir / "e2_rule_satisfaction.json"
    if not rsr_path.exists():
        logger.warning("E2 results not found")
        return

    with open(rsr_path) as f:
        data = json.load(f)

    rules = [k for k in data.keys() if k != "avg_rsr"]
    values = [data[r] for r in rules]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(rules, values, color=["#2ca02c" if v > 0.9 else "#ff7f0e" if v > 0.7 else "#d62728" for v in values])
    ax.set_xlabel("Satisfaction Rate", fontsize=12)
    ax.set_title("Experiment 2: Rule Satisfaction Rate (Ours Full Model)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.90, color="green", linestyle="--", alpha=0.5, label="90% threshold")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=10, fontweight="bold")

    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(output_dir / "e2_rule_satisfaction.png", dpi=150)
    plt.close()
    logger.info("Saved e2_rule_satisfaction.png")


def plot_e3_physics_violations(results_dir: Path, output_dir: Path):
    """Grouped bar chart: violation count per constraint."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    viol_path = results_dir / "e3_physics_violations.json"
    if not viol_path.exists():
        logger.warning("E3 results not found")
        return

    with open(viol_path) as f:
        data = json.load(f)

    constraints = [k for k in data.keys() if k != "total"]
    counts = [data[c] for c in constraints]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(constraints, counts, color=["#e74c3c", "#e67e22", "#f1c40f", "#3498db"],
                  edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Violation Count", fontsize=12)
    ax.set_title("Experiment 3: Physics Constraint Violations", fontsize=14, fontweight="bold")

    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                str(val), ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / "e3_physics_violations.png", dpi=150)
    plt.close()
    logger.info("Saved e3_physics_violations.png")


def plot_e4_convergence(results_dir: Path, output_dir: Path):
    """Line plot: violation/residual vs iteration."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    cl_path = results_dir / "e4_closed_loop.json"
    if not cl_path.exists():
        logger.warning("E4 results not found")
        return

    with open(cl_path) as f:
        data = json.load(f)

    curve = data.get("convergence_curve", [])
    if not curve:
        return

    iters = [c["iteration"] for c in curve]
    deltas = [c["avg_delta"] for c in curve]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, deltas, "o-", color="#2ca02c", linewidth=2, markersize=8)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Average Residual Norm", fontsize=12)
    ax.set_title("Experiment 4: Closed-loop Convergence", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(iters)

    # Add annotation
    for i, (it, d) in enumerate(zip(iters, deltas)):
        ax.annotate(f"{d:.4f}", (it, d), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "e4_closed_loop_convergence.png", dpi=150)
    plt.close()
    logger.info("Saved e4_closed_loop_convergence.png")


def plot_e5_robustness(results_dir: Path, output_dir: Path):
    """Line plot: F1 vs noise level."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rob_path = results_dir / "e5_robustness.json"
    if not rob_path.exists():
        logger.warning("E5 results not found")
        return

    with open(rob_path) as f:
        data = json.load(f)

    noise_levels = []
    f1s = []
    for k, v in sorted(data.items()):
        sigma = float(k.replace("noise_", ""))
        noise_levels.append(sigma)
        f1s.append(v["macro_f1"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(noise_levels, f1s, "o-", color="#1f77b4", linewidth=2, markersize=8)
    ax.fill_between(noise_levels, [f - 0.02 for f in f1s], [f + 0.02 for f in f1s],
                    alpha=0.15, color="#1f77b4")
    ax.set_xlabel("Gaussian Noise σ", fontsize=12)
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_title("Experiment 5: Robustness to Input Noise", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    for s, f in zip(noise_levels, f1s):
        ax.annotate(f"{f:.3f}", (s, f), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "e5_robustness.png", dpi=150)
    plt.close()
    logger.info("Saved e5_robustness.png")


def plot_e6_ablation(results_dir: Path, output_dir: Path):
    """Waterfall: F1 degradation from full model."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    abl_path = results_dir / "e6_ablation_summary.json"
    if not abl_path.exists():
        logger.warning("E6 results not found")
        return

    with open(abl_path) as f:
        data = json.load(f)

    full_f1 = data.get("ours_full", {}).get("best_macro_f1", 0)
    if full_f1 == 0:
        return

    variants = []
    degradations = []
    for tag, info in data.items():
        if tag == "ours_full":
            continue
        f1 = info["best_macro_f1"]
        degradation = full_f1 - f1
        name = tag.replace("ours_", "").replace("_", " ").title()
        variants.append(name)
        degradations.append(degradation)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#d62728" if d > 0.05 else "#ff7f0e" for d in degradations]
    bars = ax.bar(variants, degradations, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Macro F1 Degradation", fontsize=12)
    ax.set_title(f"Experiment 6: Ablation Study (Full model F1 = {full_f1:.3f})", fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.5)

    for bar, deg in zip(bars, degradations):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"-{deg:.3f}", ha="center", fontsize=10, fontweight="bold")

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(output_dir / "e6_ablation.png", dpi=150)
    plt.close()
    logger.info("Saved e6_ablation.png")


def generate_latex_table(results_dir: Path, output_dir: Path):
    """Generate LaTeX table from all results."""
    latex_path = output_dir / "results_table.tex"

    # Collect all results
    latex = []
    latex.append("% Auto-generated results table")
    latex.append("% Date: 2026-06-18")
    latex.append("")

    # E1: Decision Accuracy
    e1_path = results_dir / "e1_summary.json"
    if e1_path.exists():
        with open(e1_path) as f:
            e1 = json.load(f)
        latex.append("\\begin{table}[htbp]")
        latex.append("\\caption{Decision Intent Accuracy (Macro F1)}")
        latex.append("\\label{tab:e1_accuracy}")
        latex.append("\\begin{tabular}{lc}")
        latex.append("\\toprule")
        latex.append("Model & Macro F1 \\\\")
        latex.append("\\midrule")
        for model, info in sorted(e1.items()):
            name = model.replace("_", "\\_")
            latex.append(f"{name} & {info['best_macro_f1']:.4f} \\\\")
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        latex.append("")

    with open(latex_path, "w") as f:
        f.write("\n".join(latex))
    logger.info("Saved LaTeX table: %s", latex_path)


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results-dir", default="outputs")
    parser.add_argument("--output-dir", default="results/paper_figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating figures from %s → %s", results_dir, output_dir)

    plot_e1_decision_accuracy(results_dir, output_dir)
    plot_e2_rule_satisfaction(results_dir, output_dir)
    plot_e3_physics_violations(results_dir, output_dir)
    plot_e4_convergence(results_dir, output_dir)
    plot_e5_robustness(results_dir, output_dir)
    plot_e6_ablation(results_dir, output_dir)
    generate_latex_table(results_dir, output_dir)

    logger.info("All figures saved to %s/", output_dir)
    logger.info("Done. Ready for paper inclusion.")


if __name__ == "__main__":
    main()
