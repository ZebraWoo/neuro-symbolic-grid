#!/usr/bin/env python3
"""
Generate paper method figures (Figures 1-5) for:
"A Neuro-symbolic Closed-loop Learning Framework for
 Intelligent Power Grid Decision Support"

These figures do NOT require experimental data — they illustrate
the framework architecture, rule layer, closed-loop mechanism,
operational scenarios, and expert rule graph.

Usage:
  python experiments/plot_paper_figures.py --output-dir outputs/paper_figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, ConnectionPatch
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

IEEE_WIDTH = 7.16  # inches (IEEE single column)
IEEE_FULLWIDTH = 14.72  # inches (IEEE double column)

COLORS = {
    "data":      "#4472C4",  # blue - data
    "encoding":  "#5B9BD5",  # light blue - spike encoding
    "snn":       "#ED7D31",  # orange - SNN
    "symbolic":  "#70AD47",  # green - symbolic rules
    "physics":   "#A5A5A5",  # gray - physics
    "feedback":  "#FFC000",  # gold - closed-loop
    "risk":      "#FF6B6B",  # red - risk output
    "decision":  "#7030A0",  # purple - decision support
    "text":      "#333333",  # dark text
    "bg_light":  "#F7F7F7",
    "border":    "#CCCCCC",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rounded_box(ax, xy, width, height, color, text="", text_color="white",
                fontsize=8, fontweight="bold", edge_color=None, linewidth=1.5,
                alpha=1.0, zorder=2):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor=edge_color or color,
        linewidth=linewidth, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(box)
    if text:
        ax.text(
            xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, color=text_color, zorder=zorder + 1,
        )


def arrow(ax, x1, y1, x2, y2, color="#666666", linewidth=1.0, zorder=1, style="simple"):
    """Draw an arrow between two points."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->" if style == "simple" else "-|>",
            color=color, lw=linewidth,
            connectionstyle="arc3,rad=0",
        ),
        zorder=zorder,
    )


def module(ax, x, y, w, h, color, label, sublabel="", fontsize=7.5, zorder=2):
    """Draw a module block with optional sublabel."""
    rounded_box(ax, (x, y), w, h, color, "", zorder=zorder)
    ax.text(x + w/2, y + h/2 + 0.02, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white", zorder=zorder+1)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.14, sublabel, ha="center", va="center",
                fontsize=fontsize-1.5, color="white", alpha=0.85, zorder=zorder+1)


# ---------------------------------------------------------------------------
# Figure 1: Overall Framework
# ---------------------------------------------------------------------------

def draw_figure1_overall_framework(output_dir: Path):
    """FIGURE 1: Overall architecture of the proposed framework."""
    fig, ax = plt.subplots(1, 1, figsize=(IEEE_WIDTH, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor("white")

    # Title
    ax.text(5, 9.6, "PSML Multi-modal Data Streams", ha="center", va="center",
            fontsize=11, fontweight="bold", color=COLORS["text"])

    # --- Row 1: PSML Data boxes ---
    data_y = 8.6
    dw, dh = 2.2, 0.55
    data_boxes = [
        (0.5, "Load\n$P_t^L, Q_t^L$", COLORS["data"]),
        (3.9, "Renewable\n$P_t^{wind}, P_t^{solar}$", COLORS["data"]),
        (7.3, "Weather\n$T_t, W_t^{wind}, H_t$", COLORS["data"]),
    ]
    for x, label, color in data_boxes:
        rounded_box(ax, (x, data_y), dw, dh, color, label, fontsize=7, linewidth=1.2,
                    edge_color=color, alpha=0.9)

    # Arrows from data to spike encoding
    for x, _, _ in data_boxes:
        arrow(ax, x + dw/2, data_y, 5, 7.95, COLORS["encoding"], 1.2, zorder=1)

    # --- Row 2: Spike Encoding ---
    enc_y = 7.45
    module(ax, 2.5, enc_y, 5.0, 0.5, COLORS["encoding"],
           "Rate Coding: $s_i(t) = \\mathbb{1}[u_i < \\phi(x_i(t))]$",
           fontsize=7, zorder=3)

    arrow(ax, 5, enc_y, 5, 6.95, COLORS["snn"], 1.2)

    # --- Row 3: SNN Backbone (big box) ---
    snn_y = 4.8
    snn_w, snn_h = 6.8, 2.15
    # Light background box for SNN
    snn_bg = FancyBboxPatch(
        (1.6, snn_y), snn_w, snn_h,
        boxstyle="round,pad=0.12",
        facecolor="#FFF3E0", edgecolor=COLORS["snn"], linewidth=2.0,
        alpha=0.5, zorder=0,
    )
    ax.add_patch(snn_bg)
    ax.text(5, snn_y + snn_h - 0.22, "Single-compartment LIF Spiking Neural Network",
            ha="center", fontsize=9, fontweight="bold", color=COLORS["snn"],
            zorder=2)

    # Inside SNN box: membrane dynamics + blocks
    # LIF formula
    ax.text(5, 6.35,
            "$v_t = \\beta \\cdot v_{t-1} (1 - h_{t-1}) + I_t$     "
            "$h_t = H(v_t - v_{th})$",
            ha="center", fontsize=7.5, color="#555555",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor=COLORS["snn"], alpha=0.6),
            zorder=2)

    # SpikeFormer Blocks
    for i, (bx, label) in enumerate(zip([2.6, 4.0, 5.4, 6.8], ["Block 1", "Block 2", "Block 3", "Block 4"])):
        rounded_box(ax, (bx - 0.45, 5.45), 0.9, 0.35, COLORS["snn"], label,
                    fontsize=6.5, alpha=0.85, linewidth=1.0, zorder=2)

    # Dynamic representation
    rounded_box(ax, (3.2, snn_y + 0.15), 3.6, 0.35,
                "#E8C876", "$\\mathbf{z}_t$ Dynamic Grid Representation",
                text_color="#444444", fontsize=7, linewidth=1.2, zorder=2)

    # --- Branches: Symbolic & Physics ---
    # Left: Symbolic
    arrow(ax, 4.5, snn_y + 0.15, 2.0, 3.9, COLORS["symbolic"], 1.3)
    module(ax, 0.15, 3.3, 3.2, 0.6, COLORS["symbolic"],
           "Neuro-symbolic Rule Layer",
           "$\\mathcal{R}_1\\ldots\\mathcal{R}_5 \\rightarrow \\mathcal{L}_{rule}$",
           fontsize=7, zorder=3)

    # Right: Physics
    arrow(ax, 5.5, snn_y + 0.15, 8.0, 3.9, COLORS["physics"], 1.3)
    module(ax, 6.65, 3.3, 3.2, 0.6, COLORS["physics"],
           "Physics-Constrained Layer",
           "$\\mathcal{L}_{balance} + \\mathcal{L}_{ramp} + \\mathcal{L}_{capacity}$",
           fontsize=6.5, zorder=3)

    # --- Closed-loop ---
    arrow(ax, 1.8, 3.3, 4.5, 2.65, COLORS["feedback"], 1.3)
    arrow(ax, 8.2, 3.3, 5.5, 2.65, COLORS["feedback"], 1.3)

    # Unsupervised Closed-loop box
    module(ax, 2.8, 2.0, 4.4, 0.6, COLORS["feedback"],
           "Unsupervised Closed-loop Refinement",
           "$r_t = \\mathcal{L}_{rule} + \\mathcal{L}_{phy} \\rightarrow I_t^{fb} = W_f \\cdot r_t$",
           fontsize=6.5, zorder=3)

    # Feedback arrow going back up
    arrow(ax, 3.3, 2.6, 2.0, 5.7, COLORS["feedback"], 1.0)

    # --- GORS Output ---
    arrow(ax, 5, 2.0, 5, 1.3, COLORS["risk"], 1.4)
    module(ax, 3.2, 0.7, 3.6, 0.6, COLORS["risk"],
           "Grid Operational Risk Score",
           "$\\hat{y}_t \\in [0, 1]$",
           fontsize=7.5, zorder=3)

    # Decision Support
    arrow(ax, 5, 0.7, 5, 0.2, COLORS["decision"], 1.2)
    rounded_box(ax, (3.5, -0.15), 3.0, 0.35, COLORS["decision"],
                "Decision Support for Operators", fontsize=7,
                linewidth=1.2, zorder=2)

    # --- Legend / annotations on the right ---
    ax.text(9.6, 9.0, "(1) Multi-modal\n     Spike Encoding", fontsize=6.5, color=COLORS["encoding"], fontweight="bold")
    ax.text(9.6, 7.6, "(2) Spiking Temporal\n     Representation", fontsize=6.5, color=COLORS["snn"], fontweight="bold")
    ax.text(9.6, 5.8, "(3) Knowledge-guided\n     Constraints", fontsize=6.5, color=COLORS["symbolic"], fontweight="bold")
    ax.text(9.6, 4.2, "(4) Iterative Self-\n     Refinement", fontsize=6.5, color=COLORS["feedback"], fontweight="bold")
    ax.text(9.6, 2.4, "(5) Interpretable\n     Risk Score", fontsize=6.5, color=COLORS["risk"], fontweight="bold")

    # Save
    out = output_dir / "fig1_overall_framework.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.1,
                facecolor="white", edgecolor="none")
    # Also save PNG
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight",
                pad_inches=0.1, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Figure 1 saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Figure 2: Neuro-symbolic Rule Layer
# ---------------------------------------------------------------------------

def draw_figure2_neuro_symbolic_layer(output_dir: Path):
    """FIGURE 2: Differentiable neuro-symbolic rule layer mechanism."""
    fig, ax = plt.subplots(1, 1, figsize=(IEEE_WIDTH, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor("white")

    # Title
    ax.text(5, 9.6, "Differentiable Neuro-symbolic Rule Layer", ha="center",
            fontsize=11, fontweight="bold", color=COLORS["text"])

    # Left column: Expert Rule Generator
    left_x = 1.2
    rules = [
        ("$\\mathcal{R}_1$", "High Temperature\n→ Elevated Weather Risk", COLORS["data"]),
        ("$\\mathcal{R}_2$", "High Wind Speed\n→ Elevated Weather Risk", COLORS["data"]),
        ("$\\mathcal{R}_3$", "Large Net Load Δ\n→ Imbalance Risk", COLORS["data"]),
        ("$\\mathcal{R}_4$", "Large Renewable Δ\n→ Volatility Risk", COLORS["data"]),
        ("$\\mathcal{R}_5$", "Combined Extremes\n→ Systemic Risk", COLORS["data"]),
    ]
    for i, (rid, rdesc, color) in enumerate(rules):
        y = 8.4 - i * 1.05
        rounded_box(ax, (left_x - 0.3, y), 1.6, 0.85, color, f"{rid}\n{rdesc}",
                    fontsize=5.5, linewidth=1.0, alpha=0.85, zorder=2)

    # Expert Target Generator
    module(ax, 3.35, 4.8, 2.2, 1.5, COLORS["symbolic"],
           "Expert Target\nGenerator",
           "$T_k = \\sigma(\\gamma \\cdot g_k(x))$",
           fontsize=7, zorder=3)
    arrow(ax, 3.05, 7.5, 4.45, 6.5, "#999999", 0.8)
    arrow(ax, 3.05, 7.0, 4.45, 6.0, "#999999", 0.8)
    arrow(ax, 3.05, 6.0, 4.45, 5.5, "#999999", 0.8)
    arrow(ax, 3.05, 5.0, 4.45, 5.0, "#999999", 0.8)
    arrow(ax, 3.05, 4.0, 4.45, 4.5, "#999999", 0.8)

    # Right column: SNN Prediction + Gap
    arrow(ax, 5.55, 5.55, 6.8, 5.55, COLORS["snn"], 1.2)

    # SNN Feature box
    module(ax, 6.8, 4.8, 2.2, 1.5, COLORS["snn"],
           "SNN Dynamic\nFeature $z_k$",
           fontsize=8, zorder=3)

    # Side arrow from SNN
    arrow(ax, 5.0, 6.0, 6.8, 6.2, COLORS["snn"], 1.0)

    # Gap computation
    module(ax, 3.8, 2.5, 3.0, 1.2, "#E8C876",
           "Symbolic Gap\n$\\delta_k = |z_k - T_k|$",
           fontsize=7.5, zorder=3)

    arrow(ax, 4.45, 4.8, 5.3, 3.7, "#999999", 1.0)
    arrow(ax, 7.9, 4.8, 5.3, 3.7, "#999999", 1.0)

    # Soft Truth
    module(ax, 3.5, 1.05, 3.6, 0.85, COLORS["symbolic"],
           "Soft Truth Value\n$T_k^{soft} = 1/(1 + \\exp(\\tau \\cdot \\delta_k))$",
           fontsize=6.5, zorder=3)

    arrow(ax, 5.3, 2.5, 5.3, 1.9, COLORS["symbolic"], 1.2)

    # Product T-norm + Final Loss
    module(ax, 3.5, -0.05, 3.6, 0.75, COLORS["risk"],
           "Rule Trust: $\\mathcal{T} = \\prod_k T_k^{soft}$\n"
           "$\\mathcal{L}_{rule} = -\\ln(\\mathcal{T})$",
           fontsize=6.5, zorder=3)

    arrow(ax, 5.3, 1.05, 5.3, 0.7, COLORS["risk"], 1.2)

    # Annotation
    ax.text(9.5, 8.5, "(1) Heuristic rules ->\n    Continuous targets",
            fontsize=6.5, color=COLORS["symbolic"], fontweight="bold")
    ax.text(9.5, 6.5, "(2) SNN features ->\n    Dynamic representation",
            fontsize=6.5, color=COLORS["snn"], fontweight="bold")
    ax.text(9.5, 4.0, "(3) Deviation -> Soft truth\n    via sigmoid scaling",
            fontsize=6.5, color="#E8C876", fontweight="bold")
    ax.text(9.5, 1.8, "(4) Product T-norm ->\n    Differentiable penalty",
            fontsize=6.5, color=COLORS["risk"], fontweight="bold")

    out = output_dir / "fig2_neuro_symbolic_layer.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.1,
                facecolor="white", edgecolor="none")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight",
                pad_inches=0.1, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Figure 2 saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Figure 3: Physics-constrained Closed-loop
# ---------------------------------------------------------------------------

def draw_figure3_physics_closed_loop(output_dir: Path):
    """FIGURE 3: Physics constraints + closed-loop feedback mechanism."""
    fig, ax = plt.subplots(1, 1, figsize=(IEEE_WIDTH, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor("white")

    # Title
    ax.text(5, 9.6, "Physics-constrained Learning with Unsupervised Closed-loop Refinement",
            ha="center", fontsize=10.5, fontweight="bold", color=COLORS["text"])

    # --- Left: Physics constraints ---
    phys_x = 0.3
    phys_y = 7.0
    module(ax, phys_x, phys_y, 4.5, 0.6, COLORS["physics"],
           "Physics Constraints", fontsize=9, zorder=3)

    constraints = [
        ("Power Balance", "$\\mathcal{L}_{bal} = \\|P_{gen} - P_{load} - P_{loss}\\|^2$"),
        ("Ramp Rate", "$\\mathcal{L}_{ramp} = \\max(0, |\\Delta P| - R_{max})^2$"),
        ("Capacity Limit", "$\\mathcal{L}_{cap} = \\max(0, P_{ren} - P_{cap})^2$"),
    ]
    for i, (name, eq) in enumerate(constraints):
        cy = phys_y - 1.1 - i * 1.1
        rounded_box(ax, (phys_x + 0.3, cy), 3.9, 0.85,
                    "#E8E8E8", f"{name}\n{eq}",
                    text_color="#444444", fontsize=6, linewidth=1.0,
                    edge_color=COLORS["physics"], zorder=2)

    arrow(ax, 2.55, 7.0, 2.55, 6.6, COLORS["physics"], 1.0)
    arrow(ax, 2.55, 5.8, 2.55, 5.4, COLORS["physics"], 1.0)
    arrow(ax, 2.55, 4.7, 2.55, 4.3, COLORS["physics"], 1.0)

    # Physics Loss
    module(ax, phys_x + 0.5, 2.5, 3.5, 0.65, COLORS["physics"],
           "$\\mathcal{L}_{phy} = \\mathcal{L}_{bal} + \\mathcal{L}_{ramp} + \\mathcal{L}_{cap}$",
           fontsize=6.5, zorder=3)
    arrow(ax, 2.55, 3.6, 2.55, 3.15, COLORS["physics"], 1.0)

    # --- Center: SNN Prediction ---
    module(ax, 3.2, 8.5, 3.6, 0.7, COLORS["snn"],
           "SNN Prediction $\\hat{y}_t$", fontsize=9, zorder=3)

    # --- Right: Closed-loop ---
    # Consistency residual
    module(ax, 5.8, 6.5, 3.8, 1.2, "#E8C876",
           "Consistency Residual\n$r_t = (1 - \\mathcal{T}) + \\mathcal{L}_{phy}$",
           fontsize=7.5, zorder=3)

    arrow(ax, 5.0, 8.85, 7.7, 7.7, "#999999", 1.0)
    arrow(ax, 4.55, 3.15, 6.5, 6.5, COLORS["physics"], 1.0)

    # Feedback current
    module(ax, 5.8, 4.5, 3.8, 1.2, COLORS["feedback"],
           "Feedback Current\n$I_t^{fb} = W_f \\cdot \\min(r_t,\\, r_{max})$",
           fontsize=7.5, zorder=3)

    arrow(ax, 7.7, 6.5, 7.7, 5.7, COLORS["feedback"], 1.2)

    # Inject back to SNN
    module(ax, 5.8, 2.5, 3.8, 1.2, COLORS["feedback"],
           "Membrane Injection\n$v_{t+1} \\leftarrow v_{t+1} + I_t^{fb}$",
           fontsize=7.5, zorder=3)

    arrow(ax, 7.7, 4.5, 7.7, 3.7, COLORS["feedback"], 1.2)

    # Feedback loop arrow (going back to SNN)
    ax.annotate(
        "", xy=(3.5, 9.2), xytext=(9.5, 2.5),
        arrowprops=dict(
            arrowstyle="->", color=COLORS["feedback"], lw=2.0,
            connectionstyle="arc3,rad=-0.45",
            linestyle="dashed",
        ),
        zorder=1,
    )
    ax.text(6.9, 1.5, "Iterative Self-correction\n(Loop: 1–5 iterations)",
            fontsize=7, color=COLORS["feedback"], fontweight="bold",
            ha="center")

    # Refined prediction
    module(ax, 3.2, 1.2, 3.6, 0.7, COLORS["risk"],
           "Refined GORS $\\hat{y}_t^* \\in [0,1]$", fontsize=9, zorder=3)
    arrow(ax, 5.0, 3.7, 5.0, 1.9, COLORS["risk"], 1.2)
    arrow(ax, 5.0, 8.5, 5.0, 8.0, "#999999", 1.0)

    # Annotation
    ax.text(0.5, 1.0, "(1) Physical laws enforced as differentiable penalties", fontsize=6.5, color=COLORS["physics"])
    ax.text(0.5, 0.6, "(2) Internal consistency residual drives self-correction", fontsize=6.5, color=COLORS["feedback"])
    ax.text(0.5, 0.2, "(3) No ground-truth labels required for refinement", fontsize=6.5, color=COLORS["feedback"])

    out = output_dir / "fig3_physics_closed_loop.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.1,
                facecolor="white", edgecolor="none")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight",
                pad_inches=0.1, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Figure 3 saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Figure 4: Operational Scenario Analysis
# ---------------------------------------------------------------------------

def draw_figure4_operational_scenarios(output_dir: Path):
    """FIGURE 4: Three prototypical grid operational scenarios."""
    fig, axes = plt.subplots(1, 3, figsize=(IEEE_WIDTH, 3.8))
    fig.subplots_adjust(wspace=0.35)

    scenarios = [
        {
            "title": "Scenario 1: Benign Operation",
            "subtitle": "GORS ∈ [0.15, 0.35]",
            "color": COLORS["symbolic"],  # green
            "load": "Stable", "wind": "Moderate", "weather": "Mild",
            "risk_bar": 0.25, "risk_label": "Low",
            "rules": [0.95, 0.93, 0.94, 0.96, 0.91],
            "phys_penalty": 0.0,
            "fb_iter": 1,
        },
        {
            "title": "Scenario 2: Transient Ramp",
            "subtitle": "GORS ∈ [0.55, 0.70]",
            "color": COLORS["feedback"],  # orange/gold
            "load": "Increasing", "wind": "Surging +40%", "weather": "Fluctuating",
            "risk_bar": 0.62, "risk_label": "Medium",
            "rules": [0.82, 0.78, 0.72, 0.65, 0.85],
            "phys_penalty": 0.03,
            "fb_iter": 2,
        },
        {
            "title": "Scenario 3: Extreme Weather",
            "subtitle": "GORS ∈ [0.85, 0.92]",
            "color": COLORS["risk"],  # red
            "load": "Highly volatile", "wind": ">25 m/s (Typhoon)", "weather": "ΔT > 15°C",
            "risk_bar": 0.88, "risk_label": "High",
            "rules": [0.55, 0.48, 0.51, 0.42, 0.58],
            "phys_penalty": 0.12,
            "fb_iter": 4,
        },
    ]

    for idx, (ax_i, sc) in enumerate(zip(axes, scenarios)):
        ax_i.set_xlim(0, 10)
        ax_i.set_ylim(0, 12)
        ax_i.axis("off")
        ax_i.set_facecolor("white")

        # Scenario title
        ax_i.text(5, 11.6, sc["title"], ha="center", fontsize=8.5,
                  fontweight="bold", color=sc["color"])
        ax_i.text(5, 11.1, sc["subtitle"], ha="center", fontsize=7,
                  color="#888888")

        # Conditions
        bx = 1.5
        items = [
            ("Load", sc["load"]),
            ("Wind", sc["wind"]),
            ("Weather", sc["weather"]),
        ]
        for i, (label, val) in enumerate(items):
            y = 10.0 - i * 0.65
            rounded_box(ax_i, (bx, y - 0.2), 7.0, 0.55, "#F0F0F0",
                        f"{label}: {val}", text_color="#444444", fontsize=6.5,
                        edge_color="#DDDDDD", linewidth=0.8, zorder=2)

        # Risk bar (horizontal)
        bar_y = 8.0
        bar_w, bar_h = 7.0, 0.6
        # Background bar
        gradient = np.linspace(0, 1, 100).reshape(1, -1)
        ax_i.imshow(gradient, extent=[bx, bx + bar_w, bar_y, bar_y + bar_h],
                    aspect="auto", cmap="RdYlGn_r", alpha=0.5, zorder=1)
        ax_i.add_patch(plt.Rectangle((bx, bar_y), bar_w, bar_h,
                                      fill=False, edgecolor="#AAAAAA", linewidth=1.0))
        # Risk marker
        marker_x = bx + sc["risk_bar"] * bar_w
        ax_i.plot([marker_x, marker_x], [bar_y - 0.15, bar_y + bar_h + 0.15],
                  color=sc["color"], lw=3, zorder=3)
        ax_i.plot(marker_x, bar_y + bar_h / 2, 'o', color=sc["color"],
                  markersize=10, zorder=4)
        ax_i.text(marker_x, bar_y + bar_h + 0.4, f"GORS={sc['risk_bar']:.2f}",
                  ha="center", fontsize=7.5, fontweight="bold", color=sc["color"])
        ax_i.text(bx + bar_w, bar_y + bar_h + 0.4, "1.0", fontsize=6, color="#888888")
        ax_i.text(bx, bar_y + bar_h + 0.4, "0.0", fontsize=6, color="#888888")

        # Rule Trust bars
        rules_y = 6.2
        ax_i.text(bx + 0.1, rules_y + 0.6, "Rule Trust", fontsize=7,
                  fontweight="bold", color="#444444")
        for j, rt in enumerate(sc["rules"]):
            ry = rules_y - j * 0.5
            # Background
            ax_i.add_patch(plt.Rectangle((bx, ry), 5.0, 0.35,
                                          fill=False, edgecolor="#DDDDDD", linewidth=0.5))
            # Fill
            bar_color = COLORS["symbolic"] if rt > 0.7 else COLORS["feedback"] if rt > 0.5 else COLORS["risk"]
            ax_i.add_patch(plt.Rectangle((bx, ry), 5.0 * rt, 0.35,
                                          facecolor=bar_color, alpha=0.7))
            ax_i.text(bx + 5.2, ry + 0.17, f"$R_{j+1}$={rt:.2f}",
                      fontsize=5.5, color="#666666", va="center")

        # Physics & Feedback info
        info_y = 2.8
        rounded_box(ax_i, (bx, info_y), 7.0, 0.6, "#F5F5F5",
                    f"Physics Penalty: {sc['phys_penalty']:.3f}   |   "
                    f"Closed-loop Iterations: {sc['fb_iter']}",
                    text_color="#555555", fontsize=6.5, edge_color="#DDDDDD", linewidth=0.8)
        # Risk badge
        badge_x = 3.0
        rounded_box(ax_i, (badge_x, 1.0), 4.0, 0.7,
                    sc["color"], sc["risk_label"] + " Risk",
                    fontsize=10, linewidth=1.5)

    out = output_dir / "fig4_operational_scenarios.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.1,
                facecolor="white", edgecolor="none")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight",
                pad_inches=0.1, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Figure 4 saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Figure 5: Expert Rule Knowledge Graph
# ---------------------------------------------------------------------------

def draw_figure5_expert_rule_graph(output_dir: Path):
    """FIGURE 5: Expert rule knowledge graph linking environmental factors to risk."""
    fig, ax = plt.subplots(1, 1, figsize=(IEEE_WIDTH, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor("white")

    # Title
    ax.text(5, 9.7, "Expert Rule Knowledge Graph for Grid Operational Risk Assessment",
            ha="center", fontsize=10.5, fontweight="bold", color=COLORS["text"])

    # --- Input layer (environmental factors) ---
    inputs = [
        (1.0, 8.0, "Temperature\n$T_t$", COLORS["data"]),
        (3.0, 8.0, "Wind Speed\n$W_t^{wind}$", COLORS["data"]),
        (5.0, 8.0, "Net Load Δ\n$\\Delta L_t^{net}$", COLORS["data"]),
        (7.0, 8.0, "Renewable Δ\n$\\Delta R_t$", COLORS["data"]),
        (9.0, 8.0, "Solar Zenith\n$\\theta_t$", COLORS["data"]),
    ]
    input_nodes = {}
    for x, y, label, color in inputs:
        rounded_box(ax, (x - 0.65, y - 0.35), 1.3, 0.7, color, label,
                    fontsize=6, linewidth=1.2, zorder=3)
        input_nodes[label] = (x, y)

    # --- Hidden layer (intermediate risk concepts) ---
    hidden = [
        (2.0, 5.0, "Weather\nStress", "#70AD47"),
        (5.0, 5.0, "Load-Gen\nImbalance", "#70AD47"),
        (8.0, 5.0, "Renewable\nVolatility", "#70AD47"),
        (5.0, 3.0, "Systemic\nRisk", "#E8C876"),
    ]
    hidden_nodes = {}
    for x, y, label, color in hidden:
        rounded_box(ax, (x - 0.7, y - 0.35), 1.4, 0.7, color, label,
                    fontsize=6.5, linewidth=1.2, zorder=3)
        hidden_nodes[label] = (x, y)

    # --- Output layer ---
    rounded_box(ax, (3.5, 1.0), 3.0, 0.8, COLORS["risk"],
                "Grid Operational\nRisk Score (GORS)", fontsize=8, linewidth=2.0, zorder=3)

    # --- Edges ---
    edges = [
        # Temperature → Weather Stress
        (1.0, 7.65, 2.0, 5.35, "R1: High T → stress", COLORS["symbolic"]),
        # Wind Speed → Weather Stress
        (3.0, 7.65, 2.0, 5.35, "R2: Wind → stress", COLORS["symbolic"]),
        # Net Load → Load-Gen Imbalance
        (5.0, 7.65, 5.0, 5.35, "R3: Load Δ → imbalance", COLORS["symbolic"]),
        # Renewable → Renewable Volatility
        (7.0, 7.65, 8.0, 5.35, "R4: Renewable Δ → volatility", COLORS["symbolic"]),
        # Solar → Renewable Volatility
        (9.0, 7.65, 8.0, 5.35, "", COLORS["symbolic"]),
        # Weather Stress → Systemic Risk
        (2.0, 4.65, 5.0, 3.35, "", COLORS["symbolic"]),
        # Load-Gen Imbalance → Systemic Risk
        (5.0, 4.65, 5.0, 3.35, "", COLORS["symbolic"]),
        # Renewable Volatility → Systemic Risk
        (8.0, 4.65, 5.0, 3.35, "R5: Combined → risk", COLORS["symbolic"]),
        # Systemic Risk → GORS
        (5.0, 2.65, 5.0, 1.8, "", COLORS["risk"]),
    ]

    for x1, y1, x2, y2, elabel, ecolor in edges:
        style = "-|>" if elabel else "->"
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle=style, color=ecolor, lw=1.3,
                connectionstyle="arc3,rad=0",
            ),
            zorder=1,
        )
        if elabel:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.15, my, elabel, fontsize=5.5, color=ecolor,
                    fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                              edgecolor="none", alpha=0.8))

    # Legend
    ax.text(0.5, 0.3, "● Environmental Factors", fontsize=6, color=COLORS["data"], fontweight="bold")
    ax.text(3.2, 0.3, "● Risk Concepts", fontsize=6, color="#70AD47", fontweight="bold")
    ax.text(6.0, 0.3, "● Systemic Risk", fontsize=6, color="#E8C876", fontweight="bold")
    ax.text(8.8, 0.3, "● GORS Output", fontsize=6, color=COLORS["risk"], fontweight="bold")

    out = output_dir / "fig5_expert_rule_graph.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.1,
                facecolor="white", edgecolor="none")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight",
                pad_inches=0.1, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Figure 5 saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate paper method figures")
    parser.add_argument("--output-dir", default="outputs/paper_figures",
                        help="Output directory for figures")
    parser.add_argument("--figures", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                        help="Which figures to generate (default: all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating paper figures → {output_dir}/")
    print("=" * 50)

    fig_fns = {
        1: ("Overall Framework", draw_figure1_overall_framework),
        2: ("Neuro-symbolic Rule Layer", draw_figure2_neuro_symbolic_layer),
        3: ("Physics-constrained Closed-loop", draw_figure3_physics_closed_loop),
        4: ("Operational Scenarios", draw_figure4_operational_scenarios),
        5: ("Expert Rule Knowledge Graph", draw_figure5_expert_rule_graph),
    }

    for fig_num in args.figures:
        if fig_num in fig_fns:
            name, fn = fig_fns[fig_num]
            print(f"\n  Figure {fig_num}: {name}")
            fn(output_dir)

    print(f"\n{'=' * 50}")
    print(f"Done! {len(args.figures)} figures saved to {output_dir}/")
    print(f"  PDF (vector): *.pdf")
    print(f"  PNG (raster): *.png")


if __name__ == "__main__":
    main()
