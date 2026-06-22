# GORS: Neuro-symbolic Closed-loop Learning for Power Grid Decision Support

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](txt/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Official implementation of **"A Neuro-symbolic Closed-loop Learning Framework for Intelligent Power Grid Decision Support"**.

## Overview

GORS (Grid Operational Risk Score) is a neuro-symbolic framework that maps multi-modal grid observations (load, renewable generation, weather) to a unified risk score `y ∈ [0,1]`. Unlike black-box predictors, GORS provides:

- **Symbolic interpretability**: 5 differentiable expert rules with traceable truth values
- **Physical compliance**: power balance, ramp rate, and capacity constraints enforced via loss
- **Closed-loop self-correction**: consistency residuals drive feedback without ground-truth labels

## Architecture

```
PSML Input (load, renewable, irradiance, weather)
       │
       ▼
Multi-modal Spike Encoding (rate coding)
       │
       ▼
Single-compartment LIF SNN (leaky integration + spike + reset)
       │
       ▼
SpikeFormer Blocks ×4 (self-attention + LIF gating)
       │
       ▼
      ┌─ Decision Head → GORS ∈ [0,1]
      │
      ├─ Symbolic Rule Layer → L_rule = λ·(1-T)²  (5 rules, mean t-norm)
      ├─ Physics Constraints → L_phy              (balance + ramp + capacity)
      └─ Closed-loop Feedback → r_t = r_rule + r_phy → I_fb → soma
```

## Key Results

| Model | RMSE ↓ | Rule Trust ↑ | Physics Viol ↓ |
|-------|--------|-------------|----------------|
| Transformer | **0.044** | 0.179 | 0.533 |
| LSTM | 0.060 | 0.184 | 0.507 |
| TCN | 0.062 | 0.183 | 0.512 |
| SNN-LIF | 0.071 | 0.176 | 0.524 |
| **GORS** | 0.353 | **0.302** | **0.080** |

GORS improves rule trust by **68%** and reduces physics violations by **85%**. Under extreme weather, GORS degrades only **0.002 RMSE** vs. **0.026–0.038** for baselines (13× more robust).

## Quick Start

### Install

```bash
pip install torch numpy pandas scipy matplotlib
```

### Data

Download the [PSML dataset](https://github.com/tamu-engineering-research/Open-source-power-dataset) and set the path in `experiments/gors_config.py`:

```python
data_root: str = "/path/to/PSML/Minute-level Load and Renewable"
```

### Train

```bash
# Full GORS model
python experiments/train_gors.py --epochs 50 --batch-size 32 --tag gors_full

# Baselines
python experiments/train_baselines_gors.py --model lstm --epochs 50
python experiments/train_baselines_gors.py --model transformer --epochs 50
python experiments/train_baselines_gors.py --model tcn --epochs 50
python experiments/train_baselines_gors.py --model snn_lif --epochs 50

# Ablations
python experiments/train_gors.py --no-symbolic --tag gors_no_sym
python experiments/train_gors.py --no-physics --tag gors_no_phy
python experiments/train_gors.py --no-feedback --tag gors_no_fb
```

### Evaluate

```bash
# Cross-model comparison (Table 1)
python experiments/eval_cross_model.py

# Robustness (Table 4)
python experiments/eval_robustness.py

# Interpretability figure (Section IV.G)
python experiments/plot_case_study.py

# Paper figures
python experiments/plot_results.py
```

## Project Structure

```
├── experiments/           # All GORS code
│   ├── gors_config.py     # Configuration
│   ├── gors_label.py      # GORS pseudo-risk label generation
│   ├── train_gors.py      # Main training script
│   ├── train_baselines_gors.py  # Baseline training
│   ├── train_all.py       # Launch all experiments
│   ├── eval_cross_model.py     # Cross-model evaluation
│   ├── eval_robustness.py      # Robustness evaluation
│   ├── plot_case_study.py      # Interpretability figure
│   ├── plot_results.py         # Paper figures
│   ├── watch_progress.sh       # Training progress monitor
│   ├── models/            # Model architectures
│   │   ├── neuro_symbolic_model.py  # GORS backbone
│   │   ├── multi_comp_lif.py        # Multi-compartment LIF (ablation)
│   │   ├── baseline_models.py       # Baseline architectures
│   │   └── decision_head.py         # Output heads
│   ├── losses/            # Loss functions
│   │   ├── gors_symbolic_loss.py    # 5 differentiable expert rules
│   │   ├── gors_physics_loss.py     # Physical constraint penalties
│   │   └── closed_loop_loss.py      # Closed-loop feedback
│   └── trainers/          # Training infrastructure
│       └── base_trainer.py
├── src/                   # Data utilities (from PSML pipeline)
│   ├── data/              # Data loading
│   ├── control/           # Neuron models
│   ├── models/            # Model utilities
│   └── operators/         # Spike operators
├── checkpoints/           # Trained model weights
├── outputs/               # Training histories & evaluation results
├── results/               # Paper figures
│   └── paper_figures/
├── logs/                  # Training logs
├── txt/                   # Paper PDF & supplementary
└── archive/               # Deprecated code (pre-train, control, demo)
```

## Citation



