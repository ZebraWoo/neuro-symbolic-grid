#!/usr/bin/env python3
"""
Validate synthetic data quality against source data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


ALIASES: Dict[str, List[str]] = {
    "timestamp": ["时间", "采集时间", "日期时间", "时间戳", "datetime", "timestamp"],
    "feeder_id": ["台区", "台区名称", "馈线", "线路", "设备ID", "设备编号"],
    "load_kw": ["负荷", "有功功率", "负荷功率", "总有功", "P", "load_kw"],
    "pv_kw": ["光伏", "光伏出力", "光伏功率", "分布式电源出力", "pv_kw"],
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


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    q = np.linspace(0, 1, 1001)
    aq = np.quantile(a_sorted, q)
    bq = np.quantile(b_sorted, q)
    return float(np.mean(np.abs(aq - bq)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True, help="Real source CSV/XLSX file or folder")
    parser.add_argument("--synthetic", required=True, help="Synthetic CSV file or folder")
    parser.add_argument("--out", required=True, help="Validation report JSON output")
    args = parser.parse_args()

    real_path = Path(args.real).expanduser().resolve()
    syn_path = Path(args.synthetic).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    real_df, real_files = read_tables(real_path)
    real_df, real_map = normalize_columns(real_df)
    syn_df, syn_files = read_tables(syn_path)

    if "load_kw" not in real_df.columns or "load_kw" not in syn_df.columns:
        raise ValueError("Both real and synthetic data must contain load_kw.")

    real_load = pd.to_numeric(real_df["load_kw"], errors="coerce").dropna().to_numpy(dtype=float)
    syn_load = pd.to_numeric(syn_df["load_kw"], errors="coerce").dropna().to_numpy(dtype=float)

    report = {
        "real_file": str(real_path),
        "synthetic_file": str(syn_path),
        "real_rows": int(real_df.shape[0]),
        "synthetic_rows": int(syn_df.shape[0]),
        "real_files_count": len(real_files),
        "synthetic_files_count": len(syn_files),
        "column_mapping_real": real_map,
        "metrics": {
            "load_mean_real": float(np.mean(real_load)) if len(real_load) else None,
            "load_mean_syn": float(np.mean(syn_load)) if len(syn_load) else None,
            "load_std_real": float(np.std(real_load)) if len(real_load) else None,
            "load_std_syn": float(np.std(syn_load)) if len(syn_load) else None,
            "load_wasserstein": wasserstein_1d(real_load, syn_load),
        },
    }

    if "pv_kw" in real_df.columns and "pv_kw" in syn_df.columns:
        real_pv = pd.to_numeric(real_df["pv_kw"], errors="coerce").dropna().to_numpy(dtype=float)
        syn_pv = pd.to_numeric(syn_df["pv_kw"], errors="coerce").dropna().to_numpy(dtype=float)
        report["metrics"]["pv_mean_real"] = float(np.mean(real_pv)) if len(real_pv) else None
        report["metrics"]["pv_mean_syn"] = float(np.mean(syn_pv)) if len(syn_pv) else None
        report["metrics"]["pv_wasserstein"] = wasserstein_1d(real_pv, syn_pv)

    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Validation report saved: {out_path}")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
