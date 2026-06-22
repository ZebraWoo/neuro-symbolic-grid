#!/usr/bin/env python3
"""
Generate synthetic grid data from Chinese-header source data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


ALIASES: Dict[str, List[str]] = {
    "timestamp": ["时间", "采集时间", "日期时间", "时间戳", "datetime", "timestamp"],
    "feeder_id": ["台区", "台区名称", "馈线", "线路", "设备ID", "设备编号"],
    "load_kw": ["负荷", "有功功率", "负荷功率", "总有功", "P", "load_kw"],
    "pv_kw": ["光伏", "光伏出力", "光伏功率", "分布式电源出力", "pv_kw"],
    "voltage": ["电压", "电压值", "A相电压", "母线电压", "U", "voltage"],
}


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    raise ValueError(f"Unsupported file type: {suffix}")


def collect_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            [
                p
                for p in input_path.rglob("*")
                if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls"}
            ]
        )
    raise ValueError(f"Input path does not exist: {input_path}")


def read_tables(input_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    files = collect_files(input_path)
    if not files:
        raise ValueError(f"No CSV/XLSX files found under: {input_path}")
    frames: List[pd.DataFrame] = []
    for file_path in files:
        temp = read_table(file_path)
        temp["__source_file"] = str(file_path)
        frames.append(temp)
    return pd.concat(frames, ignore_index=True), [str(p) for p in files]


def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    col_map: Dict[str, str] = {}
    raw_cols = [str(c) for c in df.columns]
    for std_name, candidates in ALIASES.items():
        for c in candidates:
            if c in raw_cols:
                col_map[c] = std_name
                break
    return df.rename(columns=col_map).copy(), col_map


def sample_load(values: np.ndarray, n: int) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(n)
    base = np.random.choice(values, size=n, replace=True)
    sigma = max(float(np.std(values)) * 0.03, 1e-6)
    noise = np.random.normal(0.0, sigma, size=n)
    return np.maximum(base + noise, 0.0)


def sample_pv(hour: int, hist_day_pv: np.ndarray) -> float:
    if hour < 6 or hour > 18:
        return 0.0
    if len(hist_day_pv) == 0:
        return 0.0
    base = float(np.random.choice(hist_day_pv))
    weather = float(np.random.uniform(0.6, 1.1))
    return max(base * weather, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV/XLSX source file or folder path")
    parser.add_argument("--output", required=True, help="Output synthetic CSV")
    parser.add_argument("--samples-per-hour", type=int, default=30, help="Rows per feeder-hour")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df, source_files = read_tables(in_path)
    df_norm, used_map = normalize_columns(df)

    required = ["timestamp", "feeder_id", "load_kw"]
    missing = [c for c in required if c not in df_norm.columns]
    if missing:
        raise ValueError(
            f"Missing required fields after mapping: {missing}. "
            f"Please add Chinese aliases in ALIASES."
        )

    df_norm["timestamp"] = pd.to_datetime(df_norm["timestamp"], errors="coerce")
    df_norm["load_kw"] = pd.to_numeric(df_norm["load_kw"], errors="coerce")
    if "pv_kw" in df_norm.columns:
        df_norm["pv_kw"] = pd.to_numeric(df_norm["pv_kw"], errors="coerce")
    if "voltage" in df_norm.columns:
        df_norm["voltage"] = pd.to_numeric(df_norm["voltage"], errors="coerce")

    df_norm = df_norm.dropna(subset=["timestamp", "feeder_id", "load_kw"])
    df_norm["hour"] = df_norm["timestamp"].dt.hour

    rows = []
    for feeder_id, sub in df_norm.groupby("feeder_id"):
        for hour, g in sub.groupby("hour"):
            values = g["load_kw"].dropna().to_numpy(dtype=float)
            if len(values) < 5:
                continue
            load_samples = sample_load(values, args.samples_per_hour)
            pv_hist = (
                sub[(sub["hour"] >= 6) & (sub["hour"] <= 18)]["pv_kw"].dropna().to_numpy(dtype=float)
                if "pv_kw" in sub.columns
                else np.array([])
            )

            for load_kw in load_samples:
                pv_kw = sample_pv(int(hour), pv_hist)
                # Simple physical proxy: higher load tends to lower voltage.
                v_pu = float(np.clip(1.0 - 0.0008 * load_kw + np.random.normal(0.0, 0.005), 0.90, 1.10))
                rows.append(
                    {
                        "feeder_id": feeder_id,
                        "hour": int(hour),
                        "load_kw": float(load_kw),
                        "pv_kw": float(pv_kw),
                        "voltage_pu": v_pu,
                    }
                )

    syn = pd.DataFrame(rows)
    syn.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Synthetic rows: {len(syn)}")
    print(f"Saved to: {out_path}")
    print(f"Read source files: {len(source_files)}")
    print(f"Column mapping used: {used_map}")


if __name__ == "__main__":
    main()
