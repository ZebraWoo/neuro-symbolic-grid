# TechDoc Unified Framework Demo — Real PSML Data

三合一统一框架: **SNN-MLP 异常检测 + 多模态负荷预测 + 神经符号电网控制**

在真实 PSML 分钟级数据上联合训练三个分支。

## 与上季度合成数据版本的区别

| 维度 | 合成数据版 (techdoc_multimodal_demo_bundle) | **真实数据版 (本 bundle)** |
|------|:---:|:---:|
| 数据源 | 周期函数 + 随机扰动 | PSML 真实分钟级 CSV (CAISO/ERCOT/MISO...) |
| SNN 分支 | 合成 11-d 序列 + 规则标签 | 真实 PSML 11 维特征 + z-score 归一化 |
| 多模态分支 | 合成 4 模态 (正弦波) | 真实 PSML 列映射: load(1)+ren(2)+irr(4)+weather(4) |
| 控制分支 | 合成 3-bus 数据 | 3 个真实 PSML 区域作为 3-bus 节点 |

## 目录结构

```
techdoc_bundle/
├── README.md
├── run_demo.sh
├── requirements.txt
├── demo/
│   ├── techdoc_framework_demo.py          # 主程序 (--branch snn|multimodal|both)
│   ├── neurosymbolic_grid_demo.py         # 神经符号算子 + 真实数据采样器
│   ├── plot_techdoc_framework_demo.py     # 绘图
│   └── techdoc_results/                   # 运行后生成
└── src/
    ├── control/
    │   ├── multimodal_control_network.py  # MultimodalEmbedding
    │   └── neuron_models.py               # TemporalLIF
    └── data/
        └── load_renewable_dataset.py       # 数据加载 + 数据集分割函数
```

## 数据集分割函数

`src/data/load_renewable_dataset.py` 提供:

| 函数 | 用途 |
|------|------|
| `split_train_val_indices(n, train_ratio, seed)` | 随机分割训练/验证集索引 |
| `load_and_normalize_zone(root, zone, normalize)` | 单区域加载 + z-score/minmax 归一化 |
| `build_multimodal_windows(df, seq_len, stride, max_windows, data_fraction)` | 滑动窗口 + 4 模态拆分 + next-step target + grid state |
| `load_multi_zone_multimodal(root, zones, ...)` | 一站式: 多区域 → 多模态 → 分割 → 返回全部 |

## 环境

- Python 3.8+
- PyTorch 1.8+
- pandas, numpy, matplotlib

```bash
conda activate snn
pip install -r requirements.txt
```

## 运行

在**本目录**下执行:

```bash
# 完整框架 (推荐)
bash run_demo.sh

# 或手动指定参数
BRANCH=both EPOCHS=40 DATA_FRACTION=0.33 bash run_demo.sh

# 手动运行
python demo/techdoc_framework_demo.py --branch both --epochs 40 --data-fraction 0.33
python demo/plot_techdoc_framework_demo.py
```

### 分支说明

| --branch | 内容 |
|----------|------|
| `snn` | 11 维早期融合 SNN-MLP + CE + proximity + 符号惩罚 |
| `multimodal` | 四模态融合 + 负荷 MSE + 符号惩罚 |
| `both` | 上述全部 (默认) |

### 命令行参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--branch` | both | snn / multimodal / both |
| `--epochs` | 40 | 训练轮数 |
| `--batch-size` | 64 | 批次大小 |
| `--data-fraction` | 1.0 | 数据使用比例 (0.33=1/3) |
| `--max-windows` | 500 | 每区域最大窗口数 |
| `--zones` | (自动) | PSML 区域名列表 |
| `--data-root` | PSML 路径 | 数据根目录 |
| `--output-dir` | demo/techdoc_results | 输出目录 |

### 输出

- `demo/techdoc_results/techdoc_history.json`
- `demo/techdoc_results/techdoc_framework_curves.png`

## 联合损失 (both 分支)

```
L = L_ce + 0.5 * L_mse + 0.35 * L_penalty + 0.01 * L_smooth
```

## 版本

- 打包日期: 2026-06-04
- 数据: PSML 真实分钟级数据
