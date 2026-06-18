# 技术文档统一 Demo（多模态 + 神经符号）

在原有 `techdoc_framework_demo` 上扩展四模态融合与负荷预训练，**保留** KCL / 电压 / 潮流 / 互补四类神经符号约束与 LogicNeuron 曲线。

## 目录结构

```
techdoc_multimodal_demo_bundle/
├── README.md
├── run_demo.sh
├── requirements.txt
├── demo/
│   ├── techdoc_framework_demo.py      # 主程序（--branch snn|multimodal|both）
│   ├── plot_techdoc_framework_demo.py # 绘图
│   ├── neurosymbolic_grid_demo.py     # 神经符号算子（被 import）
│   └── techdoc_results/               # 运行后生成（示例可含历史曲线）
└── src/control/
    ├── multimodal_control_network.py  # MultimodalEmbedding
    └── neuron_models.py               # TemporalLIF（模块依赖）
```

## 环境

- Python 3.8+
- PyTorch 1.8+
- matplotlib, numpy

```bash
conda activate snn   # 或你的 PyTorch 环境
pip install -r requirements.txt
```

## 运行

在**本目录**下执行（脚本会自动 `cd` 到 bundle 根目录）：

```bash
# 完整框架：SNN-MLP + 多模态预训练 + 神经符号（推荐）
bash run_demo.sh

# 或手动
python demo/techdoc_framework_demo.py --branch both --epochs 40 --batch-size 64
python demo/plot_techdoc_framework_demo.py
```

### 分支说明

| `--branch` | 内容 |
|------------|------|
| `snn` | 11 维早期融合 SNN-MLP + CE + proximity + 符号惩罚 |
| `multimodal` | 四模态融合 + 负荷 MSE + 符号惩罚 |
| `both` | 上述全部（默认） |

### 输出

- `demo/techdoc_results/techdoc_history.json`
- `demo/techdoc_results/techdoc_framework_curves.png`

## 联合损失（both 分支）

```
L = L_ce + w_pred * L_pred + w_ctrl * L_penalty + w_smooth * L_smooth
```

- `L_pred`：下一时刻负荷 MSE（多模态融合表征）
- `L_penalty`：KCL、电压、潮流、互补（`neurosymbolic_grid_demo`）

## 与主实验关系

本 bundle 为**合成数据**可复现 demo，用于技术文档插图与符号可微验证。  
PSML 真实数据主实验见项目根目录 `train_control.py` / `outputs/psml_load_pred_curves.png`。

## 版本

- 打包日期：2026-05-20
- 对应仓库：`electric-power-regulation` demo + `src/control` 多模态嵌入
