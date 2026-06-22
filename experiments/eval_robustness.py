#!/usr/bin/env python3
"""Robustness eval: noise, missing data, extreme weather for all models."""
import sys, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.gors_config import gors_cfg
from experiments.gors_label import generate_gors_labels, GORSDataset
from experiments.train_gors import GORSModel
from experiments.train_baselines_gors import *
from src.data.multimodal_psml_dataset import load_psml_zone_frames, multimodal_psml_collate
from torch.utils.data import DataLoader

device = torch.device('cuda:0')
zones = gors_cfg.data.train_zones[:4]
zone_frames, _, _ = load_psml_zone_frames(gors_cfg.data.root, zones, max_rows_per_zone=20000, normalize="zscore")
all_data = np.concatenate([f for f in zone_frames.values()], axis=0)
windows = [all_data[i:i+96] for i in range(0, len(all_data)-96, 96)]
data = np.stack(windows)
gors = generate_gors_labels(data, 96)

# Load models
models = {}
for name, cls in [('lstm',BaselineLSTM),('transformer',BaselineTransformer),('tcn',BaselineTCN),('snn_lif',BaselineSNNLIF)]:
    m = cls().to(device).eval()
    ckpt = f'checkpoints/gors_{name}.pth'
    if Path(ckpt).exists():
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    models[name] = m

for tag in ['gors_full']:
    m = GORSModel(seq_len=96).to(device).eval()
    ckpt = f'checkpoints/{tag}.pth'
    if Path(ckpt).exists():
        ckpt_data = torch.load(ckpt, map_location=device, weights_only=False)
        sd = ckpt_data['model_state_dict'] if 'model_state_dict' in ckpt_data else ckpt_data
        m.load_state_dict(sd, strict=False)
    models[tag] = m

def perturb(modalities, mode, level):
    """Apply perturbation to modality dict."""
    result = {}
    for k, v in modalities.items():
        v_pert = v.clone()
        if mode == 'noise':
            v_pert += torch.randn_like(v) * level
        elif mode == 'missing':
            mask = torch.rand_like(v) > level
            v_pert = v_pert * mask.float()
        elif mode == 'extreme':
            if k == 'weather':
                v_pert[:, :, 1] *= 2.0  # double wind speed
            if k == 'irradiance':
                v_pert[:, :, 2] *= 0.5  # halve GHI (solar proxy)
        result[k] = v_pert
    return result

# Evaluate
results = {}
perturbations = [('clean', 'none', 0), ('noise', 'noise', 0.10), ('missing', 'missing', 0.20), ('extreme', 'extreme', 0)]
for pert_name, mode, level in perturbations:
    results[pert_name] = {}
    for model_name, model in models.items():
        ds = GORSDataset(data, gors, 96)
        loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=multimodal_psml_collate)
        total_mse, n = 0.0, 0
        with torch.no_grad():
            for modalities, targets in loader:
                mod = perturb(modalities, mode, level)
                mod = {k: v.to(device) for k, v in mod.items()}
                y = targets.to(device).unsqueeze(-1)
                if model_name.startswith('gors'):
                    pred = model(mod)['gors']
                else:
                    pred = model(flatten_input(mod).to(device))
                total_mse += ((pred - y) ** 2).sum().item()
                n += y.numel()
        rmse = np.sqrt(total_mse / max(n, 1))
        results[pert_name][model_name] = round(rmse, 4)
        print(f'{pert_name:10s} {model_name:20s}: RMSE={rmse:.4f}')

# Summary
print('\n=== Robustness Summary ===')
print(f'{"Perturbation":12s} {"LSTM":>8s} {"Transformer":>12s} {"TCN":>8s} {"SNN-LIF":>8s} {"GORS":>8s}')
clean = results['clean']
for pert_name, _, _ in perturbations:
    r = results[pert_name]
    delta = {k: r[k] - clean[k] for k in clean}
    print(f'{pert_name:12s} {delta["lstm"]:+.4f}     {delta["transformer"]:+.4f}       {delta["tcn"]:+.4f}     {delta["snn_lif"]:+.4f}     {delta["gors_full"]:+.4f}')

with open('outputs/robustness_eval.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved to outputs/robustness_eval.json')
