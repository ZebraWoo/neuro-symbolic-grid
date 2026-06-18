# Neuro-Symbolic + SNN-MLP Power-Grid Demo Bundle — Real PSML Data

两个 Demo，一份数据工具，涵盖 **脉冲神经网络异常检测** 与 **神经符号电网控制**。

## Demo 列表

| Demo | 脚本 | 功能 |
|------|------|------|
| **SNN-MLP 异常分类** | `demo/snn_mlp_demo.py` | 3 层 LIF SNN-MLP + rate coding, 在真实 PSML 11 维数据上做二分类 (正常/异常) |
| **神经符号电网控制** | `demo/neurosymbolic_grid_demo.py` | ControlNet(MLP) + 4 项可微分符号约束 (KCL/Voltage/Flow/Complement), 3-bus 系统 |

## 目录结构

```
neurosymbolic_bundle/
├── README.md
├── requirements.txt
├── run_demo.sh                    # 神经符号控制
├── run_snn_mlp.sh                 # SNN-MLP 异常分类
├── demo/
│   ├── neurosymbolic_grid_demo.py     # 神经符号主程序
│   ├── plot_neurosymbolic_demo.py     # 神经符号绘图
│   ├── snn_mlp_demo.py                # SNN-MLP 主程序
│   ├── plot_snn_mlp_demo.py           # SNN-MLP 绘图
│   ├── ns_results/                    # 神经符号输出
│   └── snn_results/                   # SNN-MLP 输出
└── src/
    └── data/
        └── load_renewable_dataset.py   # 共享数据加载 + 数据集分割函数
```

## 数据集分割函数

`src/data/load_renewable_dataset.py` 提供:

| 函数 | 用途 |
|------|------|
| `LoadRenewableDataLoader` | 扫描/加载 PSML CSV |
| `TimeSeriesDataset` | 滑动窗口数据集 |
| `split_dataset_indices(dataset, train_ratio, seed)` | 索引随机分割 |
| `multi_zone_windows(root, zones, ..., data_fraction)` | 多区域批量加载 + data_fraction 控量 |

## 环境

```bash
conda activate snn
pip install -r requirements.txt
```

## 运行

在**本目录**下执行:

### SNN-MLP 异常分类

```bash
bash run_snn_mlp.sh
# 或
python demo/snn_mlp_demo.py --epochs 12 --data-fraction 0.33
python demo/plot_snn_mlp_demo.py
```

**输出**: `demo/snn_results/` — `loss_curve.png`, `accuracy_curve.png`, `summary_report.png`

**模型**: input → fc1+LIF → fc2+LIF → fc3+LIF → rate coding → classifier(2)

### 神经符号电网控制

```bash
bash run_demo.sh
# 或
python demo/neurosymbolic_grid_demo.py --epochs 80 --data-fraction 0.33
python demo/plot_neurosymbolic_demo.py
```

**输出**: `demo/ns_results/` — `ns_penalty_curve.png`, `ns_truth_curve.png`, `summary_report.png`

**符号约束**:

| 约束 | 含义 | 目标 |
|------|------|------|
| KCL | 节点功率平衡 | gen+ren+storage−load → 0 |
| Voltage | 电压安全 | v ∈ [0.97, 1.03] |
| Flow | 联络线热稳定 | tie_flow ≤ 0.38 |
| Complement | 储能-可再生互补 | 高波动时储能平滑出力 |

## 版本

- 打包日期: 2026-06-04
- 数据: PSML 真实分钟级数据
