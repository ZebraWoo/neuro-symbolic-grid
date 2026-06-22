#!/usr/bin/env python3
"""Train baselines for GORS regression (MSE, single scalar output)."""
import sys, time, argparse, json, logging, numpy as np, torch, torch.nn as nn
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from experiments.gors_config import gors_cfg
from experiments.gors_label import generate_gors_labels, GORSDataset
from src.data.multimodal_psml_dataset import load_psml_zone_frames, multimodal_psml_collate, MODALITY_DIMS
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Models ---
class BaselineLSTM(nn.Module):
    def __init__(self, input_dim=11, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, 2, batch_first=True)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.head(out[:, -1, :]))

class BaselineTransformer(nn.Module):
    def __init__(self, input_dim=11, hidden=128):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden)
        encoder = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder, num_layers=4)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        x = self.proj(x)
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device)
        x = self.encoder(x, mask=mask)
        return torch.sigmoid(self.head(x[:, -1, :]))

class BaselineTCN(nn.Module):
    def __init__(self, input_dim=11, hidden=128):
        super().__init__()
        channels = [input_dim, 64, 128, hidden]
        self.blocks = nn.ModuleList()
        for i in range(len(channels)-1):
            in_c, out_c = channels[i], channels[i+1]
            dilation = 2**i
            self.blocks.append(nn.Sequential(
                nn.Conv1d(in_c, out_c, 3, dilation=dilation, padding=(3-1)*dilation),
                nn.BatchNorm1d(out_c), nn.ReLU(),
                nn.Conv1d(out_c, out_c, 3, dilation=dilation, padding=(3-1)*dilation),
                nn.BatchNorm1d(out_c), nn.ReLU(),
            ))
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        x = x.transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        return torch.sigmoid(self.head(x[:, :, -1]))

class BaselineSNNLIF(nn.Module):
    def __init__(self, input_dim=11, hidden=128):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden)
        self.tau, self.th = nn.Parameter(torch.tensor(2.0)), 1.0
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        B, T, D = x.shape
        out = self.proj(x)
        v = torch.zeros(B, out.shape[-1], device=x.device)
        spikes = []
        for t in range(T):
            v = v * torch.sigmoid(-1.0 / self.tau) + out[:, t, :]
            s = (v >= self.th).float()
            v = v * (1 - s)
            spikes.append(s)
        spike_seq = torch.stack(spikes, dim=1)
        rep = self.ffn(spike_seq.mean(dim=1))
        return torch.sigmoid(self.head(rep))

def flatten_input(modalities):
    tensors = []
    for name in ['load', 'renewable', 'irradiance', 'weather']:
        if name in modalities:
            tensors.append(modalities[name])
    return torch.cat(tensors, dim=-1)

MODELS = {'lstm': BaselineLSTM, 'transformer': BaselineTransformer,
          'tcn': BaselineTCN, 'snn_lif': BaselineSNNLIF}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    zones = gors_cfg.data.train_zones[:8]
    logger.info("Loading %d zones...", len(zones))
    zone_frames, _, _ = load_psml_zone_frames(gors_cfg.data.root, zones, max_rows_per_zone=args.max_rows, normalize="zscore")
    all_data = np.concatenate([f for f in zone_frames.values()], axis=0)
    windows = [all_data[i:i+96] for i in range(0, len(all_data)-96, 96)]
    data = np.stack(windows)
    gors = generate_gors_labels(data, 96)
    n = int(len(data) * 0.8)
    train_ds = GORSDataset(data[:n], gors[:n], 96)
    val_ds = GORSDataset(data[n:], gors[n:], 96)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=multimodal_psml_collate, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=multimodal_psml_collate, num_workers=2)
    logger.info("Train: %d batches, Val: %d batches", len(train_loader), len(val_loader))

    model = MODELS[args.model]().to(device)
    logger.info("Model: %.1fM params", sum(p.numel() for p in model.parameters())/1e6)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    mse = nn.MSELoss()
    best_rmse = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        for modalities, targets in train_loader:
            x = flatten_input(modalities).to(device)
            y = targets.to(device).unsqueeze(-1)
            pred = model(x)
            loss = mse(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()

        model.eval()
        total_mse, n_val = 0.0, 0
        with torch.no_grad():
            for modalities, targets in val_loader:
                x = flatten_input(modalities).to(device)
                y = targets.to(device).unsqueeze(-1)
                pred = model(x)
                total_mse += ((pred - y) ** 2).sum().item()
                n_val += y.numel()
        rmse = np.sqrt(total_mse / max(n_val, 1))

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), f"checkpoints/gors_{args.model}.pth")

        if epoch % 10 == 0 or epoch == 1:
            logger.info("Epoch %3d | Val RMSE=%.4f (best=%.4f)", epoch, rmse, best_rmse)

    logger.info("Done. Best RMSE=%.4f", best_rmse)

if __name__ == "__main__":
    main()
