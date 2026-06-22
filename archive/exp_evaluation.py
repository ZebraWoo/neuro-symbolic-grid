#!/usr/bin/env python3
"""E2-E7: Evaluation experiment stubs. Imported by run_all_experiments.sh."""

# These scripts are evaluated after training.
# Each script loads a checkpoint and runs inference on test data.

from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.config import default_cfg
from experiments.models.neuro_symbolic_model import NeuroSymbolicDecisionModel
from experiments.losses.symbolic_loss import SymbolicRuleLoss
from experiments.losses.physics_loss import PhysicsConstraintLoss
from experiments.eval_utils import (
    compute_decision_metrics, compute_rule_satisfaction,
    count_physics_violations, evaluate_closed_loop,
)
from src.data.multimodal_psml_dataset import MODALITY_DIMS, load_psml_zone_frames, multimodal_psml_collate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_test_data(data_root, zones, seq_len, stride, batch_size, max_rows, max_samples):
    """Load test data windows."""
    zone_frames, _, _ = load_psml_zone_frames(data_root, zones, max_rows_per_zone=max_rows, normalize="zscore")
    all_windows = []
    for frames in zone_frames.values():
        n = len(frames)
        n_windows = max(0, (n - seq_len) // stride + 1)
        for i in range(n_windows):
            start = i * stride
            all_windows.append(frames[start:start + seq_len])
    data = np.stack(all_windows, axis=0)
    if max_samples and max_samples < len(data):
        idx = np.random.RandomState(42).choice(len(data), max_samples, replace=False)
        data = data[idx]
    from experiments.label_decision_intents import LabeledDataset, _generate_labels_for_data
    from experiments.label_decision_intents import DecisionLabelConfig
    labels = _generate_labels_for_data(data, DecisionLabelConfig(), seq_len)
    ds = LabeledDataset(data, labels, seq_len=seq_len)
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=multimodal_psml_collate, num_workers=2)


def load_model(checkpoint_path, device):
    model = NeuroSymbolicDecisionModel(modality_dims=MODALITY_DIMS)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# E2: Rule Satisfaction Rate
# ---------------------------------------------------------------------------
def run_e2(args):
    logger.info("=== E2: Rule Satisfaction Rate ===")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    loader = load_test_data(args.data_root, default_cfg.data.test_zones[:4],
                            args.seq_len, args.stride, args.batch_size, args.max_rows, args.max_samples)

    sym_loss = SymbolicRuleLoss(temperature=20.0)
    all_rsr = {}
    with torch.no_grad():
        for modalities, labels in loader:
            modalities = {k: v.to(device) for k, v in modalities.items()}
            output = model(modalities)
            probs = torch.sigmoid(output["decision_logits"])
            features, soc = _extract_features_simple(modalities, labels, device)
            rsr = compute_rule_satisfaction(probs, features, soc)
            for k, v in rsr.items():
                all_rsr[k] = all_rsr.get(k, []) + [v]

    results = {k: float(np.mean(v)) for k, v in all_rsr.items()}
    logger.info("Rule Satisfaction Rates: %s", json.dumps(results, indent=2))
    with open(Path(args.output_dir) / "e2_rule_satisfaction.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def _extract_features_simple(modalities, labels, device):
    B = labels.shape[0]
    load = modalities.get("load", torch.zeros(B, 1, 1, device=device))
    if load.dim() == 3: load = load.mean(dim=1)
    load = load.squeeze(-1)
    renewable = modalities.get("renewable", torch.zeros(B, 1, 2, device=device))
    if renewable.dim() == 3: renewable = renewable.sum(dim=-1).mean(dim=1)
    weather = modalities.get("weather", None)
    wind_speed = weather[:, :, 1].mean(dim=1) if weather is not None and weather.dim() == 3 else torch.zeros(B, device=device)
    features = {
        "delta_renewable": torch.zeros(B, device=device),
        "delta_load": torch.zeros(B, device=device),
        "delta_power": torch.zeros(B, device=device),
        "wind_speed": wind_speed,
        "wind_threshold": torch.tensor(10.0, device=device),
    }
    soc = 0.5 + 0.3 * (labels[:, 1] - labels[:, 2])
    soc = torch.clamp(soc, 0.05, 0.95)
    return features, soc


# ---------------------------------------------------------------------------
# E3: Physics Violation Count
# ---------------------------------------------------------------------------
def run_e3(args):
    logger.info("=== E3: Physics Violation Count ===")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    loader = load_test_data(args.data_root, default_cfg.data.test_zones[:4],
                            args.seq_len, args.stride, args.batch_size, args.max_rows, args.max_samples)

    total_violations = {"ramp": 0, "soc": 0, "balance": 0, "curtail": 0, "total": 0}
    prev_probs = None
    with torch.no_grad():
        for modalities, labels in loader:
            modalities = {k: v.to(device) for k, v in modalities.items()}
            output = model(modalities)
            probs = torch.sigmoid(output["decision_logits"])
            features, soc = _extract_features_simple(modalities, labels, device)
            v = count_physics_violations(probs, prev_probs, features, soc)
            for k in total_violations:
                total_violations[k] += v[k]
            prev_probs = probs

    logger.info("Physics Violations: %s", json.dumps(total_violations, indent=2))
    with open(Path(args.output_dir) / "e3_physics_violations.json", "w") as f:
        json.dump(total_violations, f, indent=2)
    return total_violations


# ---------------------------------------------------------------------------
# E4: Closed-loop Convergence
# ---------------------------------------------------------------------------
def run_e4(args):
    logger.info("=== E4: Closed-loop Convergence ===")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    loader = load_test_data(args.data_root, default_cfg.data.test_zones[:2],
                            args.seq_len, args.stride, args.batch_size, args.max_rows,
                            min(args.max_samples or 50, 50))

    sym_loss = SymbolicRuleLoss(temperature=20.0)
    phys_loss = PhysicsConstraintLoss()

    all_histories = []
    with torch.no_grad():
        for modalities, labels in loader:
            modalities = {k: v.to(device) for k, v in modalities.items()}
            features, soc = _extract_features_simple(modalities, labels, device)
            result = evaluate_closed_loop(model, features, sym_loss, phys_loss, max_iter=5)
            all_histories.append(result["iteration_history"])

    # Average convergence curve
    if all_histories:
        max_iter = max(len(h) for h in all_histories)
        avg_curve = []
        for i in range(max_iter):
            deltas = [h[i]["delta"] for h in all_histories if i < len(h)]
            avg_curve.append({"iteration": i, "avg_delta": float(np.mean(deltas))})
        logger.info("Convergence curve: %s", json.dumps(avg_curve, indent=2))
        with open(Path(args.output_dir) / "e4_closed_loop.json", "w") as f:
            json.dump({"convergence_curve": avg_curve, "avg_iterations": float(np.mean([len(h) for h in all_histories]))}, f, indent=2)

    return {"avg_iterations": float(np.mean([len(h) for h in all_histories])) if all_histories else 0}


# ---------------------------------------------------------------------------
# E5: Robustness
# ---------------------------------------------------------------------------
def run_e5(args):
    logger.info("=== E5: Robustness ===")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    loader = load_test_data(args.data_root, default_cfg.data.test_zones[:4],
                            args.seq_len, args.stride, args.batch_size, args.max_rows, args.max_samples)

    noise_levels = [0.0, 0.05, 0.10, 0.20]
    results = {}

    for sigma in noise_levels:
        all_logits, all_labels = [], []
        with torch.no_grad():
            for modalities, labels in loader:
                modalities_noisy = {}
                for k, v in modalities.items():
                    noise = torch.randn_like(v) * sigma
                    modalities_noisy[k] = v + noise
                modalities_noisy = {k: v.to(device) for k, v in modalities_noisy.items()}
                labels = labels.to(device)
                output = model(modalities_noisy)
                all_logits.append(output["decision_logits"].cpu())
                all_labels.append(labels.cpu())
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        metrics = compute_decision_metrics(logits, labels, default_cfg.model.intent_names)
        results[f"noise_{sigma}"] = metrics
        logger.info("Noise σ=%.2f: Macro F1=%.4f", sigma, metrics["macro_f1"])

    with open(Path(args.output_dir) / "e5_robustness.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ---------------------------------------------------------------------------
# E7: Case Study (Interpretability)
# ---------------------------------------------------------------------------
def run_e7(args):
    logger.info("=== E7: Case Study ===")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)

    # Load a single day's data
    zones = default_cfg.data.test_zones[:1]
    zone_frames, _, _ = load_psml_zone_frames(
        args.data_root, zones, max_rows_per_zone=2000, normalize="zscore")
    frames = list(zone_frames.values())[0]
    data = frames[:args.seq_len * 24]  # ~1 day at 96-step windows

    windows = []
    for i in range(0, len(data) - args.seq_len, args.stride):
        windows.append(data[i:i + args.seq_len])
    if not windows:
        logger.warning("Not enough data for case study")
        return

    windows = np.stack(windows, axis=0)
    from experiments.label_decision_intents import LabeledDataset, _generate_labels_for_data
    from experiments.label_decision_intents import DecisionLabelConfig
    labels = _generate_labels_for_data(windows, DecisionLabelConfig(), args.seq_len)
    ds = LabeledDataset(windows, labels, seq_len=args.seq_len)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=multimodal_psml_collate, num_workers=2)

    case_results = []
    with torch.no_grad():
        for modalities, labels_batch in loader:
            modalities = {k: v.to(device) for k, v in modalities.items()}
            output = model(modalities)
            probs = torch.sigmoid(output["decision_logits"])
            features, soc = _extract_features_simple(modalities, labels_batch, device)
            rsr = compute_rule_satisfaction(probs, features, soc)

            for i in range(probs.shape[0]):
                case_results.append({
                    "decisions": {name: float(probs[i, j].item())
                                  for j, name in enumerate(default_cfg.model.intent_names)},
                    "rule_truths": rsr,
                    "spike_rate": float(output["spike_rate"][i].item()) if "spike_rate" in output else 0.0,
                })

    with open(Path(args.output_dir) / "e7_case_study.json", "w") as f:
        json.dump(case_results[:50], f, indent=2)  # save first 50 timepoints

    logger.info("Case study saved: %d timepoints", min(len(case_results), 50))
    return {"num_timepoints": len(case_results)}


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, choices=["e2", "e3", "e4", "e5", "e7"])
    parser.add_argument("--checkpoint", default="checkpoints/ours_full_best.pth")
    parser.add_argument("--data-root", default=default_cfg.data.root)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    runners = {"e2": run_e2, "e3": run_e3, "e4": run_e4, "e5": run_e5, "e7": run_e7}
    runners[args.exp](args)


if __name__ == "__main__":
    main()
