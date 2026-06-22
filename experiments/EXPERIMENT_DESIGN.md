# Experiment Design: A Neuro-symbolic Closed-loop Learning Framework for Intelligent Power Grid Decision Support

> **Paper frame**: Decision Support (NOT prediction, NOT regulation)
> **Core output**: Five decision intents — Increase Generation, Charge ESS, Discharge ESS, Voltage Support, Risk Warning
> **Date**: 2026-06-18

---

## 0. Experiment Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Experiment Dependency Graph                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5        │
│  Baseline     Neuro-       Closed-      Ablation    Robustness    │
│  Models       Symbolic     Loop         Study       & Case Study  │
│               Integration  Convergence                           │
│                                                                   │
│  E1: Decision Accuracy (all baselines vs Ours)                    │
│  E2: Rule Satisfaction Rate (core neuro-symbolic claim)           │
│  E3: Physics Violation Count                                      │
│  E4: Closed-loop Convergence                                      │
│  E5: Robustness (noise, missing data, extreme weather)            │
│  E6: Ablation Study (5 variants)                                  │
│  E7: Interpretability Case Study                                  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Data Configuration

### 1.1 Data Source & Features

| Data | Features | Dim | Coding |
|------|----------|-----|--------|
| Load | load_power | 1 | Rate Coding |
| Renewable | wind_power, solar_power | 2 | Rate Coding |
| Irradiance | DHI, DNI, GHI, Solar Zenith Angle | 4 | Temporal Coding |
| Weather | Dew Point, Wind Speed, Relative Humidity, Temperature | 4 | Temporal Coding |

### 1.2 Data Split

```
Train:   zones [ERCOT_1..4, MISO_1..4, SPP_1..8, PJM_1..10]  (26 zones, ~60%)
Val:     zones [CAISO_1..4, NYISO_1..6]                         (10 zones, ~20%)
Test:    zones [ERCOT_5..8, MISO_5..6, SPP_9..17, PJM_11..20]   (30 zones, ~20%)
```

### 1.3 Sliding Window

```yaml
seq_len: 96      # 96 minutes (1.6 hours)
stride: 96       # non-overlapping windows
target: next-step decision intent (derived from t+1 data)
```

---

## 2. Decision Intent Definition

### 2.1 Five Decision Intents (Multi-label Classification)

From the data at time `t`, the model outputs 5 binary intents for time `t+1`:

| Intent | Label Derivation Rule | Signal |
|--------|----------------------|--------|
| **Increase Generation** | `Δnet_load > +δ` where `net_load = load - renewable` | Net load rises → need more generation |
| **Charge ESS** | `Δrenewable > +δ AND SOC < SOC_max` | Renewable surges → store excess |
| **Discharge ESS** | `Δrenewable < -δ AND SOC > SOC_min` | Renewable drops → release stored |
| **Voltage Support** | `Δload > +2δ OR Δrenewable < -2δ` | Large net load swing → voltage risk |
| **Risk Warning** | `wind_speed > threshold OR Δload > 3δ` | Extreme weather or load event |

Where `δ = 0.1 * std(net_load)` per zone.

### 2.2 Label Generation Script

```python
# experiments/label_decision_intents.py
def generate_decision_labels(dataset, delta=0.1):
    """
    For each window, look at the difference between window_average and
    next_window_average to derive decision intents.
    Returns: Tensor[B, 5] of binary labels.
    """
```

---

## 3. Experiment Matrix

### E1: Decision Intent Accuracy

**Goal**: Show that Ours outperforms baselines on decision intent classification.

**Baselines**:
| ID | Model | Description | Script |
|----|-------|-------------|--------|
| B1 | LSTM | 2-layer LSTM, hidden=128 | `exp_baselines.py --model lstm` |
| B2 | Transformer | 4-layer Transformer encoder | `exp_baselines.py --model transformer` |
| B3 | TCN | Temporal ConvNet, 4 layers | `exp_baselines.py --model tcn` |
| B4 | SNN-LIF | Single-compartment LIF SNN | `exp_baselines.py --model snn_lif` |
| B5 | SNN-Izh | Single-compartment Izhikevich SNN | `exp_baselines.py --model snn_izh` |
| **Ours** | Multi-comp SNN + Neuro-symbolic + Closed-loop | Full framework | `exp_ours.py` |

**Metrics**: Per-class F1, Macro F1, Micro F1, Hamming Loss

**Expected output table**:
```
Model              | IncGen F1 | ChgESS F1 | DisESS F1 | VoltSup F1 | RiskWarn F1 | Macro F1
LSTM               | 0.72      | 0.68      | 0.65      | 0.70       | 0.67        | 0.684
Transformer        | 0.76      | 0.72      | 0.70      | 0.74       | 0.71        | 0.726
TCN                | 0.74      | 0.70      | 0.68      | 0.72       | 0.69        | 0.706
SNN-LIF            | 0.78      | 0.74      | 0.72      | 0.77       | 0.73        | 0.748
SNN-Izh            | 0.80      | 0.76      | 0.74      | 0.79       | 0.76        | 0.770
Ours (full)        | 0.89      | 0.87      | 0.85      | 0.88       | 0.86        | 0.870
```

**Run command**:
```bash
# All baselines
python experiments/exp_e1_decision_accuracy.py --model all --epochs 50 --batch-size 32

# Ours
python experiments/exp_ours_full.py --epochs 50 --batch-size 32
```

---

### E2: Rule Satisfaction Rate (Core Neuro-symbolic Experiment)

**Goal**: Prove that the neuro-symbolic layer makes decisions that satisfy logical rules.

**Rules evaluated**:

| Rule ID | Condition | Expected Intent | Soft Truth Formula |
|---------|-----------|-----------------|-------------------|
| R1 | ΔRenewable > δ | Charge ESS | `σ(temp * (storage_action - expert_target))` |
| R2 | ΔLoad > δ | Increase Generation | `σ(temp * (gen_action - load_signal))` |
| R3 | Wind Speed > τ | Risk Warning | `σ(temp * (risk - wind_anomaly))` |
| R4 | SOC > SOC_max | NOT Charge ESS | `σ(temp * (SOC_max - SOC))` |
| R5 | SOC < SOC_min | NOT Discharge ESS | `σ(temp * (SOC - SOC_min))` |
| R6 | |ΔP| > ramp_limit | Voltage Support | `σ(temp * (ramp_limit - |ΔP|))` |

**Metric**: Rule Satisfaction Rate (RSR) = fraction of samples where soft_truth > 0.5 for applicable rules.

**Expected output table**:
```
Method               | R1(%) | R2(%) | R3(%) | R4(%) | R5(%) | R6(%) | Avg RSR
LSTM                 | 68    | 72    | 65    | 78    | 76    | 70    | 71.5
Transformer          | 74    | 76    | 70    | 82    | 80    | 74    | 76.0
SNN-LIF              | 78    | 79    | 74    | 85    | 83    | 77    | 79.3
Ours w/o Symbolic    | 82    | 83    | 78    | 88    | 86    | 81    | 83.0
Ours (full)          | 95    | 94    | 91    | 98    | 97    | 93    | 94.7
```

**Run command**:
```bash
python experiments/exp_e2_rule_satisfaction.py --checkpoint checkpoints/ours_full.pth
```

---

### E3: Physics Violation Count

**Goal**: Show that physics constraints reduce violations.

**Constraints** (adjusted for PSML data, no IEEE 39-bus needed):

| Constraint | Formula | Threshold |
|------------|---------|-----------|
| Ramp Rate | `|P_t - P_{t-1}|` | `R_max = 0.15 * P_rated` |
| SOC Bounds | `SOC_min < SOC < SOC_max` | `[0.1, 0.9]` |
| Power Balance | `|Gen + ESS_disch - ESS_ch - NetLoad|` | `< 0.05 * P_rated` |
| Curtailment | `Curtail_t` | `< Curtail_max = 0.2 * Renew_t` |

**Metric**: Count of violations over the entire test set.

**Expected output table**:
```
Method              | Ramp Viol. | SOC Viol. | Bal Viol. | Curtail Viol. | Total
Without Constraints| 245        | 89        | 156       | 34            | 524
Physics Only        | 52         | 18        | 41        | 12            | 123
Physics + Symbolic  | 18         | 7         | 12        | 5             | 42
Ours (full)         | 12         | 5         | 8         | 3             | 28
```

**Run command**:
```bash
python experiments/exp_e3_physics_violation.py --checkpoint checkpoints/ours_full.pth
```

---

### E4: Closed-loop Convergence

**Goal**: Show that constraint-check → residual → feedback → update loop reduces violations iteratively.

**Loop design**:
```
Decision z_0 → Constraint Check → Residual e_0 → Feedback Injection → z_1 → ...
```

**Metric**: Violation count per iteration, convergence within K iterations.

**Expected output**:
```
Iteration:  0      1      2      3      4      5
Violations: 28  →  15  →  8   →  5   →  3   →  3
RSR:        78% →  85% →  91% →  94% →  96% →  96%
```

**Visualization**: Violation-vs-iteration curve converging to near-zero.

```bash
python experiments/exp_e4_closed_loop.py --checkpoint checkpoints/ours_full.pth --max-iter 5
```

---

### E5: Robustness

**Goal**: Show degradation is graceful under distribution shift.

**Perturbation types**:

| Type | Description | Levels |
|------|-------------|--------|
| Gaussian Noise | Add N(0, σ²) to all features | σ ∈ {0.05, 0.10, 0.20} |
| Missing Data | Randomly mask p% of timesteps | p ∈ {10%, 20%, 30%} |
| Extreme Weather | Double wind_speed, halve solar | 1 scenario |

**Expected output table** (Macro F1):
```
Condition            | LSTM  | Transf.| SNN-LIF | Ours   | Δ(Ours-LSTM)
Clean (test)         | 0.684 | 0.726  | 0.748   | 0.870  | +0.186
Noise σ=0.10         | 0.612 | 0.654  | 0.691   | 0.832  | +0.220
Missing 20%          | 0.598 | 0.638  | 0.675   | 0.818  | +0.220
Extreme Weather      | 0.554 | 0.602  | 0.638   | 0.795  | +0.241
```

```bash
python experiments/exp_e5_robustness.py --checkpoint checkpoints/ours_full.pth
```

---

### E6: Ablation Study

**Goal**: Quantify the contribution of each component.

| Variant ID | Description | What's Removed |
|------------|-------------|----------------|
| **Full** | Complete framework | — |
| **A1: No Spike** | Replace SNN with MLP/Tanh | Multi-comp SNN → 3-layer MLP |
| **A2: No Symbolic** | Remove symbolic rule layer | λ_symbolic = 0 |
| **A3: No Physics** | Remove physics constraints | L_physics = 0 |
| **A4: No Closed-loop** | Single forward pass only | No feedback iteration |
| **A5: Single-comp** | Replace multi-comp with single LIF | Multi-comp → standard LIF |

**Expected output table**:
```
Variant             | Macro F1 | RSR(%)  | Physics Viol. | Train Time
Full (Ours)         | 0.870    | 94.7    | 28            | 1.00x
A1: No Spike        | 0.812    | 86.3    | 89            | 0.72x
A2: No Symbolic     | 0.835    | 83.0    | 42            | 0.95x
A3: No Physics      | 0.848    | 91.2    | 156           | 0.93x
A4: No Closed-loop  | 0.852    | 90.1    | 52            | 0.88x
A5: Single-comp SNN | 0.841    | 89.5    | 38            | 0.85x
```

```bash
python experiments/exp_e6_ablation.py --run-all
```

---

### E7: Interpretability Case Study

**Goal**: Qualitatively demonstrate WHY the model makes certain decisions.

**Scenario**: Pick one day with:
- Morning: wind ramps up
- Noon: solar peaks
- Evening: load surges, solar drops, wind drops

**Visualization outputs** (one figure per scenario):
1. **Input panel**: Load, Wind, Solar, Temperature time series
2. **Spike raster**: Multi-comp neuron spike activity over time
3. **Decision intent panel**: 5 binary signals over time
4. **Rule truth panel**: Soft truth values for each rule over time
5. **Closed-loop panel**: Violation count decreasing over iterations

```bash
python experiments/exp_e7_case_study.py --checkpoint checkpoints/ours_full.pth --date "2019-03-15"
```

---

## 4. Implementation Plan

### 4.1 New Files to Create

```
experiments/
├── EXPERIMENT_DESIGN.md          # This document
├── config.py                     # Shared configs: paths, hyperparams, rule definitions
├── label_decision_intents.py     # Decision intent label generation from PSML data
├── models/
│   ├── __init__.py
│   ├── neuro_symbolic_model.py   # Full Ours model: Multi-comp SNN + Symbolic + Physics
│   ├── baseline_models.py        # LSTM, Transformer, TCN, SNN-LIF, SNN-Izh baselines
│   └── decision_head.py          # 5-output decision intent head
├── losses/
│   ├── __init__.py
│   ├── symbolic_loss.py          # Soft logic rule loss (truth computation)
│   ├── physics_loss.py           # Ramp/SOC/Balance/Curtailment constraint loss
│   └── closed_loop_loss.py       # Feedback residual computation
├── trainers/
│   ├── __init__.py
│   ├── base_trainer.py           # Shared training loop
│   └── closed_loop_trainer.py    # Closed-loop iteration wrapper
├── exp_e1_decision_accuracy.py   # Experiment 1
├── exp_e2_rule_satisfaction.py   # Experiment 2
├── exp_e3_physics_violation.py   # Experiment 3
├── exp_e4_closed_loop.py         # Experiment 4
├── exp_e5_robustness.py          # Experiment 5
├── exp_e6_ablation.py            # Experiment 6
├── exp_e7_case_study.py          # Experiment 7
├── exp_ours_full.py              # Train Ours full model
├── eval_utils.py                 # Shared evaluation: RSR, physics violation counter, etc.
└── plot_results.py               # Generate all paper-ready figures
```

### 4.2 Modification to Existing Files

| File | Change |
|------|--------|
| `src/control/multimodal_control_network.py` | Add `num_control_outputs=5` support, add symbolic loss integration hooks |
| `src/control/advanced_neuron_models.py` | Wire Izhikevich + Multi-compartment into the training forward path |
| `train_control.py` | No change needed — experiments use separate scripts |

### 4.3 Execution Order

```
Step 1: experiments/config.py             (1h)   Shared config
Step 2: experiments/label_decision_intents.py (2h) Label generation
Step 3: experiments/models/baseline_models.py (3h) All baselines
Step 4: experiments/models/neuro_symbolic_model.py (4h) Ours model
Step 5: experiments/losses/*.py           (3h)   Symbolic + Physics + Closed-loop losses
Step 6: experiments/trainers/*.py         (2h)   Training infrastructure
Step 7: experiments/exp_ours_full.py      (2h)   Train full model
Step 8: experiments/eval_utils.py         (2h)   Evaluation infrastructure
Step 9: experiments/exp_e1_*.py           (1h)   Run + collect
Step 10: experiments/exp_e2_*.py          (1h)   Run + collect
Step 11: experiments/exp_e3_*.py          (1h)   Run + collect
Step 12: experiments/exp_e4_*.py          (1h)   Run + collect
Step 13: experiments/exp_e5_*.py          (1h)   Run + collect
Step 14: experiments/exp_e6_*.py          (2h)   Run all ablations
Step 15: experiments/exp_e7_*.py          (2h)   Case study with plots
Step 16: experiments/plot_results.py      (3h)   Paper-ready figures
```

---

## 5. Hyperparameter Configuration

```yaml
# experiments/config.py

data:
  root: "/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable"
  train_zones: ["ERCOT_zone_1_", "ERCOT_zone_2_", "ERCOT_zone_3_", "ERCOT_zone_4_",
                "MISO_zone_1_", "MISO_zone_2_", "MISO_zone_3_", "MISO_zone_4_",
                "SPP_zone_1_", "SPP_zone_2_", "SPP_zone_3_", "SPP_zone_4_",
                "SPP_zone_5_", "SPP_zone_6_", "SPP_zone_7_", "SPP_zone_8_",
                "PJM_zone_1_", "PJM_zone_2_", "PJM_zone_3_", "PJM_zone_4_",
                "PJM_zone_5_", "PJM_zone_6_", "PJM_zone_7_", "PJM_zone_8_",
                "PJM_zone_9_", "PJM_zone_10_"]
  val_zones: ["CAISO_zone_1_", "CAISO_zone_2_", "CAISO_zone_3_", "CAISO_zone_4_",
              "NYISO_zone_1_", "NYISO_zone_2_", "NYISO_zone_3_", "NYISO_zone_4_",
              "NYISO_zone_5_", "NYISO_zone_6_"]
  test_zones: ["ERCOT_zone_5_", "ERCOT_zone_6_", "ERCOT_zone_7_", "ERCOT_zone_8_",
               "MISO_zone_5_", "MISO_zone_6_",
               "SPP_zone_9_", "SPP_zone_10_", "SPP_zone_11_", "SPP_zone_12_",
               "SPP_zone_13_", "SPP_zone_14_", "SPP_zone_15_", "SPP_zone_16_",
               "SPP_zone_17_",
               "PJM_zone_11_", "PJM_zone_12_", "PJM_zone_13_", "PJM_zone_14_",
               "PJM_zone_15_", "PJM_zone_16_", "PJM_zone_17_", "PJM_zone_18_",
               "PJM_zone_19_", "PJM_zone_20_"]
  seq_len: 96
  stride: 96
  val_ratio: 0.1

model:
  embedding_dim: 32
  hidden_dim: 64
  num_blocks: 4
  num_compartments: 3          # dendrite(2) + soma(1)
  neuron_type: "RS"            # Izhikevich Regular Spiking
  num_decision_intents: 5      # IncGen, ChgESS, DisESS, VoltSup, RiskWarn
  use_lif: True
  use_multi_comp: True

symbolic:
  temperature: 20.0
  num_rules: 6
  rule_weights: [1.0, 1.0, 1.0, 0.8, 0.8, 0.6]  # R1-R6
  lambda_symbolic: 0.15

physics:
  ramp_limit: 0.15             # relative to rated power
  soc_min: 0.10
  soc_max: 0.90
  balance_tolerance: 0.05
  curtail_max: 0.20
  lambda_physics: 0.25

closed_loop:
  max_iterations: 5
  fast_lr: 0.01
  slow_lr_multiplier: 0.1
  convergence_threshold: 0.001

training:
  epochs: 50
  batch_size: 32
  lr: 0.001
  weight_decay: 1e-5
  scheduler: "cosine"
  gradient_clip: 1.0
  device: "cuda"
  num_workers: 4
  seed: 42
```

---

## 6. Expected Runtime

| Experiment | GPU Hours (A100) | Wall Clock |
|------------|------------------|------------|
| E1: All baselines (6 models × 50 epochs) | ~2h | 2h (parallel) |
| Ours full training (50 epochs) | ~1.5h | 1.5h |
| E2: Rule Satisfaction (eval only) | ~0.1h | 5min |
| E3: Physics Violation (eval only) | ~0.1h | 5min |
| E4: Closed-loop Convergence (eval only) | ~0.3h | 15min |
| E5: Robustness (3 conditions × eval) | ~0.3h | 15min |
| E6: Ablation (5 variants × 50 epochs) | ~4h | 4h (parallel) |
| E7: Case Study (eval + plot) | ~0.1h | 10min |
| **Total** | **~8.3h** | **~4h (with parallel)** |

---

## 7. Quick Start

```bash
# Step 1: Generate decision labels
python experiments/label_decision_intents.py

# Step 2: Train baselines (run in parallel)
for model in lstm transformer tcn snn_lif snn_izh; do
    python experiments/exp_e1_decision_accuracy.py --model $model --epochs 50 &
done
wait

# Step 3: Train Ours full model
python experiments/exp_ours_full.py --epochs 50

# Step 4: Run evaluation experiments (these only do inference)
python experiments/exp_e2_rule_satisfaction.py
python experiments/exp_e3_physics_violation.py
python experiments/exp_e4_closed_loop.py
python experiments/exp_e5_robustness.py
python experiments/exp_e6_ablation.py
python experiments/exp_e7_case_study.py

# Step 5: Generate paper figures
python experiments/plot_results.py --output-dir results/paper_figures/
```

---

## 8. Key Risks & Mitigations

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Decision labels too noisy | Medium | Use multi-step smoothing; label by trend direction rather than absolute threshold |
| SNN training unstable (vanishing gradients) | Medium | Use surrogate gradient from existing code (`SurrogateSpike`); clip gradients; warm-start with LIF |
| Symbolic loss dominates task loss | Low | Set `lambda_symbolic=0.15` initially; use adaptive weighting |
| Closed-loop iterations diverge | Low | Cap iterations at 5; use damped feedback (`fast_lr=0.01`) |
| Multi-comp SNN too slow | Medium | Use 2 dendrite compartments (not 3); reduce hidden_dim to 48 for ablation A5 |
| Baseline models overfit small zones | Low | Use 26 train zones; apply weight decay |

---

## 9. Success Criteria

| Criterion | Threshold |
|-----------|-----------|
| Ours Macro F1 > best baseline | +5% minimum |
| Rule Satisfaction Rate > 90% | Must exceed |
| Physics violations reduced > 50% vs no-constraint | Must exceed |
| Closed-loop converges in ≤ 5 iterations | Must satisfy |
| Robustness: Ours drop < baseline drop under perturbation | Must satisfy |
| Ablation: all removals cause degradation | Must satisfy |
| Case study: decisions are human-interpretable | Qualitative |
