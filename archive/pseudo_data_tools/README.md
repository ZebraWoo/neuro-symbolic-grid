# 中文表头伪数据工具

适用于农网/配网表格（CSV/XLSX，中文表头），支持单文件或文件夹批量读取（递归扫描子目录）：

- `01_data_profile_cn.py`：字段标准化 + 数据画像
- `02_generate_synthetic_cn.py`：生成伪数据
- `03_validate_synthetic_cn.py`：校验伪数据与真实数据分布差异

## 1) 数据画像

```bash
python experiments/pseudo_data_tools/01_data_profile_cn.py \
  --input "data/raw" \
  --out-dir "data/processed/pseudo_profile"
```

输出：

- `profile.json`
- `hourly_load_stats.csv`
- `normalized_preview.csv`

## 2) 伪数据生成

```bash
python experiments/pseudo_data_tools/02_generate_synthetic_cn.py \
  --input "data/raw" \
  --output "data/pseudo/synthetic_grid_data.csv" \
  --samples-per-hour 30 \
  --seed 42
```

## 3) 质量校验

```bash
python experiments/pseudo_data_tools/03_validate_synthetic_cn.py \
  --real "data/raw" \
  --synthetic "data/pseudo/synthetic_grid_data.csv" \
  --out "data/processed/pseudo_profile/validation_report.json"
```

## 中文字段别名

脚本内置常见别名（如 `时间/采集时间`、`负荷/有功功率`、`光伏/光伏出力`、`台区/馈线`）。

如果你的字段名不同，直接在各脚本顶部 `ALIASES` 里新增即可。

## 一键执行（.sh）

```bash
bash experiments/pseudo_data_tools/run_pseudo_pipeline.sh "data/raw"
```

可选第二个参数指定输出根目录（默认 `data`）：

```bash
bash experiments/pseudo_data_tools/run_pseudo_pipeline.sh "data/raw" "data"
```
