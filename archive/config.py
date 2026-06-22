"""
Shared configuration for all experiments.

Paper: A Neuro-symbolic Closed-loop Learning Framework
       for Intelligent Power Grid Decision Support
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

    # Train / Val / Test zone splits
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
        "SPP_zone_13_", "SPP_zone_14_", "SPP_zone_15_", "SPP_zone_16_",
        "SPP_zone_17_",
        "PJM_zone_11_", "PJM_zone_12_", "PJM_zone_13_", "PJM_zone_14_",
        "PJM_zone_15_", "PJM_zone_16_", "PJM_zone_17_", "PJM_zone_18_",
        "PJM_zone_19_", "PJM_zone_20_",
    ])

    # Sliding window
    seq_len: int = 96
    stride: int = 96

    # Feature columns (from PSML)
    feature_columns: List[str] = field(default_factory=lambda: [
        "load_power", "wind_power", "solar_power",
        "DHI", "DNI", "GHI", "Solar Zenith Angle",
        "Dew Point", "Wind Speed", "Relative Humidity", "Temperature",
    ])

    # Modality grouping (matches src/data/multimodal_psml_dataset.py)
    modality_dims: Dict[str, int] = field(default_factory=lambda: {
        "load": 1,
        "renewable": 2,
        "irradiance": 4,
        "weather": 4,
    })

    val_ratio: float = 0.1
    normalize: str = "zscore"  # "zscore" or "minmax"


# ---------------------------------------------------------------------------
# Decision Intent Labels
# ---------------------------------------------------------------------------

@dataclass
class DecisionLabelConfig:
    """Configuration for generating decision intent labels from PSML data."""
    delta_ratio: float = 0.10          # threshold = delta_ratio * std(net_load)
    delta_smoothing: int = 5           # smoothing window for trend detection
    wind_speed_threshold: float = 10.0  # m/s, for Risk Warning
    load_surge_threshold: float = 0.30  # 30% change = surge
    soc_max: float = 0.90
    soc_min: float = 0.10


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    # SNN backbone
    embedding_dim: int = 32
    hidden_dim: int = 64
    num_blocks: int = 4
    num_heads: int = 4
    dropout: float = 0.1

    # Multi-compartment neuron
    num_dendrite_compartments: int = 2  # each receives different modality groups
    soma_neuron_type: str = "RS"        # Izhikevich: RS, FS, LTS, IB
    use_lif_in_blocks: bool = True
    use_multi_comp: bool = True

    # Decision output
    num_decision_intents: int = 5       # IncGen, ChgESS, DisESS, VoltSup, RiskWarn

    @property
    def intent_names(self) -> List[str]:
        return ["Increase Generation", "Charge ESS", "Discharge ESS",
                "Voltage Support", "Risk Warning"]


# ---------------------------------------------------------------------------
# Symbolic Rules
# ---------------------------------------------------------------------------

@dataclass
class SymbolicConfig:
    temperature: float = 20.0           # sigmoid steepness for soft logic
    num_rules: int = 6

    # Rule weights in total loss
    rule_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 0.8, 0.8, 0.6])
    lambda_symbolic: float = 0.15

    # Rule definitions (human-readable)
    rule_names: List[str] = field(default_factory=lambda: [
        "R1: ΔRenewable↑ → Charge ESS",
        "R2: ΔLoad↑ → Increase Generation",
        "R3: Wind Speed↑↑ → Risk Warning",
        "R4: SOC > SOC_max → Prohibit Charge",
        "R5: SOC < SOC_min → Prohibit Discharge",
        "R6: |ΔP| > Ramp Limit → Voltage Support",
    ])


# ---------------------------------------------------------------------------
# Physics Constraints
# ---------------------------------------------------------------------------

@dataclass
class PhysicsConfig:
    ramp_limit: float = 0.15            # relative to rated power
    soc_min: float = 0.10
    soc_max: float = 0.90
    balance_tolerance: float = 0.05     # relative to rated power
    curtail_max: float = 0.20           # max curtailment fraction
    lambda_physics: float = 0.25

    # Constraint penalty weights
    w_ramp: float = 1.0
    w_soc: float = 1.0
    w_balance: float = 1.5
    w_curtail: float = 0.5


# ---------------------------------------------------------------------------
# Closed-loop
# ---------------------------------------------------------------------------

@dataclass
class ClosedLoopConfig:
    max_iterations: int = 5
    fast_lr: float = 0.01               # step size for output correction
    slow_lr_multiplier: float = 0.1     # relative to training lr
    convergence_threshold: float = 0.001


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

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    log_interval: int = 5


# ---------------------------------------------------------------------------
# Master config aggregator
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    labels: DecisionLabelConfig = field(default_factory=DecisionLabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    symbolic: SymbolicConfig = field(default_factory=SymbolicConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    closed_loop: ClosedLoopConfig = field(default_factory=ClosedLoopConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict:
        """Flatten all configs for JSON serialization."""
        result = {}
        for section in ["data", "labels", "model", "symbolic", "physics",
                        "closed_loop", "training"]:
            cfg = getattr(self, section)
            for k, v in cfg.__dict__.items():
                if isinstance(v, Path):
                    v = str(v)
                if isinstance(v, list) and len(v) > 10:
                    v = f"[{len(v)} items]"
                result[f"{section}.{k}"] = v
        return result


# Singleton
default_cfg = ExperimentConfig()
