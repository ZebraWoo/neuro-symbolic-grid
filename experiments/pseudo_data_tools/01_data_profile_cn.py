#!/usr/bin/env python3
"""
Profile raw grid data with Chinese headers.

Usage:
  python experiments/pseudo_data_tools/01_data_profile_cn.py \
    --input "data/raw/your_file.csv" \
    --out-dir "data/processed/pseudo_profile"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


ALIASES: Dict[str, List[str]] = {
    "timestamp": ["时间", "采集时间", "日期时间", "时间戳", "datetime", "timestamp"],
    "feeder_id": ["台区", "台区名称", "馈线", "线路", "设备ID", "设备编号"],
    "load_kw": ["负荷", "有功功率", "负荷功率", "总有功", "P", "load_kw"],
    "pv_kw": ["光伏", "光伏出力", "光伏功率", "分布式电源出力", "pv_kw"],
    "voltage": ["电压", "电压值", "A相电压", "母线电压", "U", "voltage"],
    "current": ["电流", "电流值", "A相电流", "I", "current"],
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
        files = sorted(
            [
                p
                for p in input_path.rglob("*")
                if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls"}
            ]
        )
        return files
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV/XLSX file path or folder path")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

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
    df_norm = df_norm.dropna(subset=["timestamp"])
    df_norm["hour"] = df_norm["timestamp"].dt.hour

    numeric_cols = [c for c in ["load_kw", "pv_kw", "voltage", "current"] if c in df_norm.columns]
    numeric_stats = {}
    for col in numeric_cols:
        s = pd.to_numeric(df_norm[col], errors="coerce").dropna()
        if s.empty:
            continue
        numeric_stats[col] = {
            "count": int(s.shape[0]),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
            "max": float(s.max()),
        }

    by_hour = (
        df_norm.groupby("hour")["load_kw"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"mean": "load_mean", "std": "load_std"})
    )

    profile = {
        "input_file": str(in_path),
        "source_files_count": len(source_files),
        "source_files": source_files[:200],
        "rows_after_cleaning": int(df_norm.shape[0]),
        "column_mapping": used_map,
        "numeric_stats": numeric_stats,
        "feeder_count": int(df_norm["feeder_id"].nunique()),
    }

    (out_dir / "profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    by_hour.to_csv(out_dir / "hourly_load_stats.csv", index=False, encoding="utf-8-sig")
    df_norm.head(2000).to_csv(out_dir / "normalized_preview.csv", index=False, encoding="utf-8-sig")

    print(f"Profile saved to: {out_dir}")
    print(f"Column mapping: {used_map}")


if __name__ == "__main__":
    main()
