"""Quick eval: compute Rule Trust + Physics Violations for all models."""
import sys, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.gors_config import gors_cfg
from experiments.gors_label import generate_gors_labels, GORSDataset
from experiments.train_gors import GORSModel
from experiments.losses.gors_symbolic_loss import GORSSymbolicLoss
from experiments.losses.gors_physics_loss import GORSPhysicsLoss
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
ds = GORSDataset(data, gors, 96)
loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=multimodal_psml_collate)

sym_fn = GORSSymbolicLoss().to(device)
phys_fn = GORSPhysicsLoss().to(device)

models = {}
# Baselines
for name, cls in [('lstm',BaselineLSTM),('transformer',BaselineTransformer),('tcn',BaselineTCN),('snn_lif',BaselineSNNLIF)]:
    m = cls().to(device).eval()
    ckpt = f'checkpoints/gors_{name}.pth'
    if Path(ckpt).exists():
        sd = torch.load(ckpt, map_location=device, weights_only=False)
        m.load_state_dict(sd)
    models[name] = m

# GORS variants (checkpoints wrap state_dict in a dict)
for tag in ['gors_full','gors_no_sym','gors_no_phy','gors_no_fb']:
    m = GORSModel(seq_len=96).to(device).eval()
    ckpt = f'checkpoints/{tag}.pth'
    if Path(ckpt).exists():
        ckpt_data = torch.load(ckpt, map_location=device, weights_only=False)
        sd = ckpt_data['model_state_dict'] if 'model_state_dict' in ckpt_data else ckpt_data
        m.load_state_dict(sd, strict=False)
    models[tag] = m

results = {}
for name, model in models.items():
    total_mse, total_trust, total_viol, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for modalities, targets in loader:
            mod = {k: v.to(device) for k, v in modalities.items()}
            y = targets.to(device).unsqueeze(-1)
            B = y.shape[0]

            if name.startswith('gors'):
                out = model(mod)
                pred = out['gors']
                _, _, sym_m = sym_fn(pred, mod)
                _, _, phys_m = phys_fn(pred, mod)
                trust = sym_m['trust_comprehensive']
                viol = phys_m.get('r_phy', 0)
            else:
                x = flatten_input(mod).to(device)
                pred = model(x)
                # Compute rule/physics on baseline predictions too
                _, _, sym_m = sym_fn(pred, mod)
                _, _, phys_m = phys_fn(pred, mod)
                trust = sym_m['trust_comprehensive']
                viol = phys_m.get('r_phy', 0)

            mse = ((pred - y) ** 2).mean().item()
            total_mse += mse * B; total_trust += trust * B; total_viol += viol * B; n += B

    rmse = np.sqrt(total_mse / n)
    results[name] = {'rmse': round(rmse, 4), 'trust': round(total_trust / n, 4), 'viol': round(total_viol / n, 4)}
    print(f'{name:20s}: RMSE={rmse:.4f}  Trust={total_trust/n:.4f}  Viol={total_viol/n:.4f}')

with open('outputs/cross_model_eval.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved to outputs/cross_model_eval.json')
