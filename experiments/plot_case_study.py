#!/usr/bin/env python3
"""Generate 4-panel interpretability figure for Section IV.G."""
import sys, json, numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.gors_config import gors_cfg
from experiments.train_gors import GORSModel
from experiments.losses.gors_symbolic_loss import GORSSymbolicLoss
from src.data.multimodal_psml_dataset import load_psml_zone_frames

device = torch.device('cuda:0')
zones = gors_cfg.data.train_zones[:2]  # 2 zones for realistic data

# Load data
zone_frames, _, _ = load_psml_zone_frames(gors_cfg.data.root, zones, max_rows_per_zone=30000, normalize="zscore")
all_data = np.concatenate([f for f in zone_frames.values()], axis=0)

# Pick a segment with wind ramp — scan for max wind speed change
windows = [all_data[i:i+96] for i in range(0, len(all_data)-96, 96)]
data = np.stack(windows)
wind_col = 8  # wind speed
wind_means = data[:, :, wind_col].mean(axis=1)
# Find segment with large wind increase
deltas = np.diff(wind_means)
ramp_start = np.argmax(deltas[:len(deltas)//2])  # in first half of data
start_idx = max(0, ramp_start - 10)
n_windows = min(24, len(data) - start_idx)  # 24 windows ≈ 24*96min/60 ≈ 38 hours, use 24 windows

segment = data[start_idx:start_idx + n_windows]
print(f"Case study: {n_windows} windows starting at idx {start_idx}, wind ramp at window {ramp_start - start_idx}")

# Load model
model = GORSModel(seq_len=96).to(device).eval()
ckpt = 'checkpoints/gors_full.pth'
ckpt_data = torch.load(ckpt, map_location=device, weights_only=False)
sd = ckpt_data['model_state_dict'] if 'model_state_dict' in ckpt_data else ckpt_data
model.load_state_dict(sd, strict=False)

sym_fn = GORSSymbolicLoss().to(device)

# Run inference window by window
gors_vals, truth_vals, spike_rates = [], [], []
load_vals, wind_gen, solar_gen, wind_speed = [], [], [], []

from experiments.gors_label import PreloadedDataset
from torch.utils.data import DataLoader
from src.data.multimodal_psml_dataset import multimodal_psml_collate

with torch.no_grad():
    for i in range(n_windows):
        w = segment[i:i+1]  # [1, 96, 11]
        ds = PreloadedDataset(w, 96)
        loader = DataLoader(ds, batch_size=1, collate_fn=multimodal_psml_collate)
        for modalities, _ in loader:
            mod = {k: v.to(device) for k, v in modalities.items()}
            out = model(mod)
            gors_vals.append(out['gors'].item())

            # Rule truths
            _, _, sym_m = sym_fn(out['gors'], mod)
            truth_vals.append([sym_m['T1_temp'], sym_m['T2_wind'], sym_m['T3_load'], sym_m['T4_ren'], sym_m['T5_sys']])

            # Spike rate
            spike_rates.append(out['spike_rate'].item() if out['spike_rate'] is not None else 0)

            # Input features
            weather = mod['weather'][0, -1, :]  # last timestep
            load_vals.append(mod['load'][0, -1, 0].item())
            wind_gen.append(mod['renewable'][0, -1, 0].item())
            solar_gen.append(mod['renewable'][0, -1, 1].item())
            wind_speed.append(weather[1].item())  # wind speed is col 1 in weather

gors_vals = np.array(gors_vals)
truth_vals = np.array(truth_vals)
spike_rates = np.array(spike_rates)

# === Plot ===
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
time_labels = [f'{i*1.6:.0f}h' for i in range(n_windows)]  # 96min ≈ 1.6h per window

# Panel 1: Input features
ax1 = axes[0]
ax1.plot(load_vals, 'b-', label='Load', linewidth=1.5, alpha=0.8)
ax1.plot(wind_gen, 'g-', label='Wind Gen', linewidth=1.5, alpha=0.8)
ax1.plot(solar_gen, 'orange', label='Solar Gen', linewidth=1.5, alpha=0.8)
ax1_twin = ax1.twinx()
ax1_twin.plot(wind_speed, 'r--', label='Wind Speed', linewidth=2, alpha=0.6)
ax1.set_ylabel('Normalized Power', fontsize=11)
ax1_twin.set_ylabel('Wind Speed (z-score)', fontsize=11, color='r')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
ax1.set_title('Panel 1: Input Features', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Panel 2: GORS
ax2 = axes[1]
colors = ['#d62728' if v > 0.7 else '#ff7f0e' if v > 0.4 else '#2ca02c' for v in gors_vals]
ax2.bar(range(n_windows), gors_vals, color=colors, edgecolor='white', width=0.8)
ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='High Risk (0.7)')
ax2.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Medium Risk (0.4)')
ax2.set_ylabel('GORS', fontsize=11)
ax2.set_title('Panel 2: Grid Operational Risk Score', fontsize=13, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9)
ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.3)

# Mark the wind ramp event
ramp_rel = ramp_start - start_idx
if 0 <= ramp_rel < n_windows:
    ax2.axvline(x=ramp_rel, color='purple', linestyle=':', linewidth=2, alpha=0.7)
    ax2.annotate('Wind Ramp\nOnset', xy=(ramp_rel, 0.85), fontsize=10, color='purple',
                 ha='center', fontweight='bold')

# Panel 3: Rule Truths
ax3 = axes[2]
rule_names = ['T1:Temp', 'T2:Wind', 'T3:Load', 'T4:Renew', 'T5:System']
colors_rules = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
for i in range(5):
    ax3.plot(truth_vals[:, i], '-o', color=colors_rules[i], label=rule_names[i], linewidth=1.5, markersize=4)
ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
ax3.set_ylabel('Soft Truth Value', fontsize=11)
ax3.set_title('Panel 3: Symbolic Rule Truth Values', fontsize=13, fontweight='bold')
ax3.legend(loc='lower left', fontsize=9, ncol=5)
ax3.set_ylim(0, 1.05)
ax3.grid(True, alpha=0.3)

# Panel 4: Spike Activity
ax4 = axes[3]
ax4.fill_between(range(n_windows), spike_rates, alpha=0.4, color='#17becf')
ax4.plot(spike_rates, 'o-', color='#17becf', linewidth=2, markersize=5)
ax4.set_ylabel('Avg Firing Rate', fontsize=11)
ax4.set_xlabel('Time Window (96 min/window)', fontsize=12)
ax4.set_title('Panel 4: SNN Spike Activity', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Mark high-risk period
if np.any(gors_vals > 0.7):
    high_risk_start = np.argmax(gors_vals > 0.7)
    ax4.axvspan(high_risk_start - 0.3, n_windows - 0.3, alpha=0.1, color='red')
    ax4.text(high_risk_start + 0.5, spike_rates.max() * 0.9, 'High Risk\nPeriod',
             fontsize=10, color='red', fontweight='bold')

plt.tight_layout()
output_path = 'results/paper_figures/fig_case_study.png'
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved: {output_path}')

# Print data for paper text
print(f'\n=== Case Study Data ===')
print(f'Peak GORS: {gors_vals.max():.3f} at window {np.argmax(gors_vals)}')
print(f'T2 (wind) min at peak: {truth_vals[np.argmax(gors_vals), 1]:.3f}')
print(f'Spike rate increase: {spike_rates[np.argmax(gors_vals)] / max(spike_rates[:ramp_rel], default=0.01):.1f}x vs pre-ramp')
print(f'Avg truth pre-ramp: {truth_vals[:ramp_rel].mean():.3f}')
print(f'Avg truth post-ramp: {truth_vals[ramp_rel:].mean():.3f}')
