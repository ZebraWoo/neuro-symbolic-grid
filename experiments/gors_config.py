"""
GORS (Grid Operational Risk Score) Framework Configuration.

Refactored per advisor feedback:
  - Single scalar output y ∈ [0,1]
  - Labels derived from data volatility (no fake intents)
  - Closed-loop driven by consistency residual r = r_rule + r_phy
  - Self-supervised: no ground-truth control decisions needed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    root: str = "/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable"

    train_zones: List[str] = field(default_factory=lambda: [
        "ERCOT_zone_1_", "ERCOT_zone_2_", "ERCOT_zone_3_", "ERCOT_zone_4_",
        "MISO_zone_1_", "MISO_zone_2_", "MISO_zone_3_", "MISO_zone_4_",
        "SPP_zone_1_", "SPP_zone_2_", "SPP_zone_3_", "SPP_zone_4_",
        "SPP_zone_5_", "SPP_zone_6_", "SPP_zone_7_", "SPP_zone_8_",
        "PJM_zone_1_", "PJM_zone_2_", "PJM_zone_3_", "PJM_zone_4_",
        "PJM_zone_5_", "PJM_zone_6_", "PJM_zone_7_", "PJM_zone_8_",
        "PJM_zone_9_", "PJM_zone_10_",
    ])

    val_zones: List[str] = field(default_factory=lambda: [
        "CAISO_zone_1_", "CAISO_zone_2_", "CAISO_zone_3_", "CAISO_zone_4_",
        "NYISO_zone_1_", "NYISO_zone_2_", "NYISO_zone_3_", "NYISO_zone_4_",
        "NYISO_zone_5_", "NYISO_zone_6_",
    ])

    test_zones: List[str] = field(default_factory=lambda: [
        "ERCOT_zone_5_", "ERCOT_zone_6_", "ERCOT_zone_7_", "ERCOT_zone_8_",
        "MISO_zone_5_", "MISO_zone_6_",
        "SPP_zone_9_", "SPP_zone_10_", "SPP_zone_11_", "SPP_zone_12_",
        "SPP_zone_13_", "SPP_zone_14_", "SPP_zone_15_", "SPP_zone_16_", "SPP_zone_17_",
        "PJM_zone_11_", "PJM_zone_12_", "PJM_zone_13_", "PJM_zone_14_",
        "PJM_zone_15_", "PJM_zone_16_", "PJM_zone_17_", "PJM_zone_18_",
        "PJM_zone_19_", "PJM_zone_20_",
    ])

    seq_len: int = 96
    stride: int = 96
    modality_dims: Dict[str, int] = field(default_factory=lambda: {
        "load": 1, "renewable": 2, "irradiance": 4, "weather": 4,
    })
    val_ratio: float = 0.1
    normalize: str = "zscore"


# ---------------------------------------------------------------------------
# GORS Pseudo-Risk Label Generation
# ---------------------------------------------------------------------------

@dataclass
class GORSLabelConfig:
    """
    Generate pseudo-risk target y ∈ [0,1] from data volatility.

    y = σ(w1 * |Δnet_load|/σ_nl + w2 * |Δrenewable|/σ_r + w3 * wind_anomaly)
    where σ is sigmoid, mapping unbounded volatility → [0,1] risk score.
    """
    w_net_load: float = 0.5     # net load volatility weight
    w_renewable: float = 0.3    # renewable volatility weight
    w_weather: float = 0.2      # weather anomaly weight
    sigmoid_scale: float = 2.0  # steepness of volatility → risk mapping


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    embedding_dim: int = 64
    hidden_dim: int = 128
    num_blocks: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    num_dendrite_compartments: int = 2
    soma_neuron_type: str = "RS"
    use_lif_in_blocks: bool = True
    use_multi_comp: bool = False  # single-comp LIF SNN (simpler, faster, better for GORS)
    # GORS: single scalar output
    output_dim: int = 1


# ---------------------------------------------------------------------------
# Symbolic Rules (Risk-based)
# ---------------------------------------------------------------------------

@dataclass
class SymbolicConfig:
    temperature: float = 20.0
    lambda_symbolic: float = 0.02  # reduced: old 0.15 dominated MSE

    # Rules evaluate whether GORS properly reflects risk factors
    # R1: High temperature → elevated weather risk
    # R2: High wind speed → elevated weather risk
    # R3: Large net load change → elevated imbalance risk
    # R4: Large renewable change → elevated volatility risk
    # R5: Combined extreme → elevated systemic risk
    num_rules: int = 5


# ---------------------------------------------------------------------------
# Physics Constraints (Risk consistency)
# ---------------------------------------------------------------------------

@dataclass
class PhysicsConfig:
    """Physical constraints on GORS consistency."""
    lambda_physics: float = 0.25
    w_balance: float = 1.5    # power balance constraint
    w_ramp: float = 1.0       # ramp rate constraint
    w_capacity: float = 0.5   # renewable capacity constraint


# ---------------------------------------------------------------------------
# Closed-loop
# ---------------------------------------------------------------------------

@dataclass
class ClosedLoopConfig:
    max_iterations: int = 5
    fast_lr: float = 0.01
    convergence_threshold: float = 0.001
    # Feedback: r_t = r_rule + r_phy → I_fb = W_f * r_t → inject into soma
    lambda_feedback: float = 0.10


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 32
    lr: float = 0.001
    weight_decay: float = 1e-5
    scheduler: str = "cosine"
    gradient_clip: float = 1.0
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    log_interval: int = 5


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass
class GORSConfig:
    data: DataConfig = field(default_factory=DataConfig)
    labels: GORSLabelConfig = field(default_factory=GORSLabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    symbolic: SymbolicConfig = field(default_factory=SymbolicConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    closed_loop: ClosedLoopConfig = field(default_factory=ClosedLoopConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


gors_cfg = GORSConfig()
