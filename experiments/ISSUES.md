# Paper Implementation Issues

> **Paper**: A Neuro-symbolic Closed-loop Learning Framework for Intelligent Power Grid Decision Support
> **Target**: 核心期刊或三大检索论文 (per 技术规范书 §3)
> **Timeline**: 2026年7月 — 2026年12月

---

## Issue 1: 决策标签生成与数据管线验证

**Status**: 🔴 待开始
**Priority**: P0 (阻塞所有后续实验)
**Estimated**: 2h
**Assignee**: 刘小娇

### 任务
- [ ] 运行 `experiments/label_decision_intents.py` 生成 train/val/test 三组决策标签
- [ ] 验证标签分布合理性（每个 intent 的 positive rate 在 10%-50% 之间）
- [ ] 如果标签严重不均衡，调整 `DecisionLabelConfig.delta_ratio` 或 `wind_speed_threshold`
- [ ] 确认 `LabeledDataset` 包装器能正确返回 `(modalities_dict, decision_labels)`

### 依赖
- 无（仅依赖 PSML 数据已就位）

### 文件
- `experiments/label_decision_intents.py`
- `experiments/config.py` → `DecisionLabelConfig`

### 验收标准
```
✅ train/val/test 三组 .npy 标签文件生成成功
✅ 每个 intent 的 positive rate 输出合理
✅ python experiments/label_decision_intents.py 不报错
```

---

## Issue 2: Baseline 模型训练 (E1)

**Status**: 🔴 待开始
**Priority**: P0
**Estimated**: 4h (含 GPU 训练时间)
**Assignee**: 刘小娇
**Depends on**: Issue 1

### 任务
- [ ] 跑通 `exp_e1_decision_accuracy.py --model lstm`
- [ ] 跑通所有 5 个 baseline: lstm, transformer, tcn, snn_lif, snn_izh
- [ ] 确认每个 baseline 的 val_macro_f1 > 0.60（否则需要调参）
- [ ] 记录每个 baseline 的训练时间、参数量、best F1
- [ ] 如 LSTM/Transformer 效果差，调 hidden_dim/lr/num_layers

### 文件
- `experiments/exp_e1_decision_accuracy.py`
- `experiments/models/baseline_models.py`
- `experiments/trainers/base_trainer.py`
- `experiments/eval_utils.py`

### 验收标准
```
✅ 5 个 baseline 均训练完成，checkpoints/ 下有对应 .pth 文件
✅ outputs/e1_summary.json 有所有 baseline 的 best_macro_f1
✅ 每个 baseline 的 Macro F1 > 0.60
```

---

## Issue 3: Ours 完整模型训练

**Status**: 🔴 待开始
**Priority**: P0 (核心实验)
**Estimated**: 4h
**Assignee**: 刘小娇
**Depends on**: Issue 1

### 任务
- [ ] 跑通 `exp_ours_full.py`（完整模型：多室SNN + 符号规则 + 物理约束 + 闭环）
- [ ] 确认训练过程中 symbolic_loss 和 physics_loss 都在下降
- [ ] 确认 val_macro_f1 > 0.80（目标比最佳 baseline 高 5%+）
- [ ] 如训练不稳定（SNN 梯度消失/爆炸），调整 `gradient_clip` 或 `surrogate gradient` 参数
- [ ] 如 symbolic_loss 不下降，调整 `lambda_symbolic` 或 `temperature`
- [ ] 记录 best checkpoint 路径 `checkpoints/ours_full_best.pth`

### 文件
- `experiments/exp_ours_full.py`
- `experiments/models/neuro_symbolic_model.py` ← 多室 Izhikevich + 脉冲编码
- `experiments/losses/symbolic_loss.py` ← 6条规则的软逻辑损失
- `experiments/losses/physics_loss.py` ← 4项物理约束损失
- `experiments/trainers/base_trainer.py`

### 验收标准
```
✅ Ours 完整模型训练完成
✅ val_macro_f1 > 0.80 且 > best baseline + 0.05
✅ train_sym 和 train_phys 在 history JSON 中呈下降趋势
✅ checkpoints/ours_full_best.pth 存在
```

---

## Issue 4: 规则满足率与物理约束评估 (E2, E3)

**Status**: 🔴 待开始
**Priority**: P1
**Estimated**: 2h
**Assignee**: 刘小娇
**Depends on**: Issue 2, Issue 3

### 任务
- [ ] 跑 E2: 对所有模型计算 6 条规则的满足率 (RSR)
- [ ] 跑 E3: 对所有模型统计物理约束违反次数
- [ ] 确认 Ours 的 avg_rsr > 90%（否则调整 symbolic loss 权重）
- [ ] 确认 Ours 的 physics violations 比 no-constraint 减少 > 50%
- [ ] 生成 E2/E3 的结果 JSON 文件

### 文件
- `experiments/exp_evaluation.py` → `run_e2()`, `run_e3()`
- `experiments/eval_utils.py` → `compute_rule_satisfaction()`, `count_physics_violations()`

### 验收标准
```
✅ outputs/e2_rule_satisfaction.json: Ours avg_rsr > 0.90
✅ outputs/e3_physics_violations.json: Ours total < baseline_w/o_constraints * 0.5
✅ 至少 3 个 baseline 的 E2/E3 结果也记录在案（用于论文对比表）
```

---

## Issue 5: 闭环收敛验证 (E4)

**Status**: 🔴 待开始
**Priority**: P1
**Estimated**: 1.5h
**Assignee**: 刘小娇
**Depends on**: Issue 3

### 任务
- [ ] 跑 E4: 加载 `ours_full_best.pth`，运行闭环迭代
- [ ] 确认 5 轮迭代内 residual norm 收敛（delta < 0.001）
- [ ] 如不收敛，调整 `fast_lr` (0.001 → 0.05) 或 `max_iterations`
- [ ] 如震荡，加入 damped feedback
- [ ] 生成收敛曲线数据

### 文件
- `experiments/exp_evaluation.py` → `run_e4()`
- `experiments/losses/closed_loop_loss.py`
- `experiments/eval_utils.py` → `evaluate_closed_loop()`

### 验收标准
```
✅ outputs/e4_closed_loop.json: 平均迭代次数 ≤ 5
✅ 收敛曲线呈现单调下降趋势
```

---

## Issue 6: 鲁棒性测试 (E5)

**Status**: 🔴 待开始
**Priority**: P1
**Estimated**: 2h
**Assignee**: 刘小娇
**Depends on**: Issue 2, Issue 3

### 任务
- [ ] 跑 E5: 对 Ours + 3 个 baseline 做噪声/缺失/极端天气鲁棒性测试
- [ ] 确认 Ours 在所有扰动下的 F1 下降幅度 < baseline 的下降幅度
- [ ] 如 Ours 鲁棒性不如预期，检查 symbolic rules 是否在扰动下给出合理的 soft constraint 信号
- [ ] 生成鲁棒性曲线数据

### 文件
- `experiments/exp_evaluation.py` → `run_e5()`

### 验收标准
```
✅ outputs/e5_robustness.json: 所有噪声级别下的 F1 数据
✅ Ours 在高噪声(σ=0.20)下的 F1 下降 < 10%（相对 clean）
✅ Ours 鲁棒性优于所有 baseline
```

---

## Issue 7: 消融实验 (E6)

**Status**: 🔴 待开始
**Priority**: P1
**Estimated**: 5h (5个变体 × ~1h 训练)
**Assignee**: 刘小娇
**Depends on**: Issue 3

### 任务
- [ ] 跑全部 5 个消融变体:
  - A1: `--no-spike` (去掉脉冲编码，用 MLP)
  - A2: `--no-symbolic` (去掉符号规则层)
  - A3: `--no-physics` (去掉物理约束)
  - A4: `--no-closed-loop` (单次前向，无闭环)
  - A5: `--no-multi-comp` (单室 LIF 替代多室 Izhikevich)
- [ ] 确认每个消融变体的 F1 < Ours_full
- [ ] 确认 "No Symbolic" 变体的 RSR 显著下降
- [ ] 确认 "No Physics" 变体的 violation count 显著上升
- [ ] 确认 "No Spike" 变体的计算效率差异
- [ ] 生成消融汇总表

### 文件
- `experiments/exp_e6_ablation.py`
- `experiments/exp_ours_full.py` (被 --no-* 参数调用)

### 验收标准
```
✅ outputs/e6_ablation_summary.json: 5 个变体 + full 的结果
✅ 所有消融变体的 F1 < Ours_full
✅ "No Symbolic" 的 RSR < Ours_full - 5%
✅ "No Physics" 的 violations > Ours_full * 2
```

---

## Issue 8: 可解释性案例研究 (E7)

**Status**: 🔴 待开始
**Priority**: P2
**Estimated**: 2h
**Assignee**: 刘小娇
**Depends on**: Issue 3

### 任务
- [ ] 选取一天典型数据（如风电骤升 + 光伏下降的天气过程）
- [ ] 跑 E7: 生成逐时间步的决策意图概率、规则真值、脉冲发放率
- [ ] 检查决策是否与规则一致（如风电上升时 Charge ESS 概率确实上升）
- [ ] 如可解释性不理想，检查规则真值是否在关键时间步有明显变化
- [ ] 导出 case study JSON 用于论文插图

### 文件
- `experiments/exp_evaluation.py` → `run_e7()`

### 验收标准
```
✅ outputs/e7_case_study.json: 至少 24 个时间步的详细决策记录
✅ 规则真值在关键事件时刻有显著性变化
✅ 脉冲发放率与决策激活性有合理关联
```

---

## Issue 9: 论文图表生成

**Status**: 🔴 待开始
**Priority**: P1
**Estimated**: 3h
**Assignee**: 刘小娇
**Depends on**: Issue 4, Issue 5, Issue 6, Issue 7, Issue 8

### 任务
- [ ] 跑 `plot_results.py` 生成全部 7 张图
- [ ] 检查每张图的清晰度、字体大小、配色
- [ ] 生成 LaTeX 结果表格
- [ ] 人工检查图表中的数字是否与 JSON 结果一致
- [ ] 如有 matplotlib 中文显示问题，配置中文字体或改用英文标签

### 文件
- `experiments/plot_results.py`

### 验收标准
```
✅ results/paper_figures/e1_decision_accuracy.png
✅ results/paper_figures/e2_rule_satisfaction.png
✅ results/paper_figures/e3_physics_violations.png
✅ results/paper_figures/e4_closed_loop_convergence.png
✅ results/paper_figures/e5_robustness.png
✅ results/paper_figures/e6_ablation.png
✅ results/paper_figures/results_table.tex
```

---

## Issue 10: 论文写作 — Section 3 (Methodology)

**Status**: 🔴 待开始
**Priority**: P0
**Estimated**: 2 周
**Assignee**: 刘小娇
**Depends on**: Issue 3 (需有完整模型实现后才能写)

### 任务
- [ ] 3.1 Spike Encoding: 速率编码+时间编码混合策略的数学描述
- [ ] 3.2 Multi-compartment SNN: 树突-胞体 Izhikevich 动力学（专利公式直接搬）
- [ ] 3.3 Symbolic Rule Layer: 6 条规则 → 专家目标 → 软逻辑真值 → t-norm 综合信任度
- [ ] 3.4 Physics Constraints: 4 项约束的数学定义
- [ ] 3.5 Closed-loop Learning: 快慢双环的数学描述 + 反馈电流注入
- [ ] 画系统总架构图（Fig. 1）

### 核心公式（专利已有）
- 速率编码: \( f_j = \frac{1}{T}\int_t^{t+T} s_j(\tau)d\tau \)
- 树突动力学: \( C_j\frac{dV_{dend,j}}{dt} = g_L(E_L-V_{dend,j}) + I_{syn,j} + g_{c,j}(V_s-V_{dend,j}) \)
- 胞体 Izhikevich: \( \frac{dV_s}{dt} = 0.04V_s^2 + 5V_s + 140 - u + \frac{1}{A_s}\sum g_{c,j}(V_{dend,j}-V_s) \)
- 软逻辑真值: \( truth_m = \sigma(temp \cdot comp\_gap_m) \)
- 综合信任度: \( trust = \prod_{m=1}^{M} truth_m \)
- 总损失: \( \mathcal{L} = \mathcal{L}_{BCE} + \mathcal{L}_{physics} + \lambda(1-trust) \)
- 快环修正: \( \hat{z} = z - \eta_{fast} \cdot \nabla_z \mathcal{L}_{physics} \)

### 验收标准
```
✅ Section 3 初稿完成，公式与专利/代码一致
✅ 系统架构图清晰表达 "感知→演化→推理→决策→闭环" 五层
```

---

## Issue 11: 论文写作 — Section 1-2 (Intro + Related Work)

**Status**: 🔴 待开始
**Priority**: P0
**Estimated**: 1 周
**Assignee**: 刘小娇
**Depends on**: 无（纯文献调研）

### 任务
- [ ] Section 1: 明确科学问题——高比例新能源下"物理不可行输出"和"黑盒不可解释"
- [ ] 列出本文 3-4 个贡献
- [ ] Section 2.1: Power Grid AI (LSTM/Transformer/RL → 指出缺乏物理约束)
- [ ] Section 2.2: SNN in Power Systems (几乎是空白，这是卖点)
- [ ] Section 2.3: Neuro-symbolic Learning (Logic Tensor Networks, DeepProbLog → 指出本工作是首次将可微符号推理与SNN结合用于电力)
- [ ] 整理 30-40 篇参考文献

### 验收标准
```
✅ Section 1-2 初稿完成
✅ 参考文献 .bib 文件至少有 30 条
```

---

## Issue 12: 论文写作 — Section 4-5 (Experiments + Conclusion)

**Status**: 🔴 待开始
**Priority**: P1
**Estimated**: 2 周
**Assignee**: 刘小娇
**Depends on**: Issue 4-9 (需实验结果)

### 任务
- [ ] 4.1 Dataset & Setup: PSML 数据集介绍、数据划分、超参配置
- [ ] 4.2 Baselines: 5 个 baseline 的简要说明
- [ ] 4.3 Main Results: 填入 E1-E4 的真实数据
- [ ] 4.4 Robustness: 填入 E5 数据
- [ ] 4.5 Ablation Study: 填入 E6 数据 + 分析每个组件的贡献
- [ ] 4.6 Interpretability: 填入 E7 案例 + 分析
- [ ] Section 5: Conclusion + Future Work

### 验收标准
```
✅ Section 4-5 初稿完成，所有数字来自真实实验
✅ 每个实验段落都有 "why this result matters" 的分析
```

---

## Issue 13: 论文定稿 + 投稿

**Status**: 🔴 待开始
**Priority**: P2
**Estimated**: 1 周
**Assignee**: 刘小娇
**Depends on**: Issue 10, 11, 12

### 任务
- [ ] 全文通读，修正 typo 和不一致
- [ ] 检查所有公式的符号一致性
- [ ] 检查所有图的清晰度和字体
- [ ] 确定目标期刊（建议：中国电机工程学报 / Applied Energy / IEEE Trans. on Smart Grid）
- [ ] 按期刊模板排版
- [ ] 请导师或同事审阅
- [ ] 投稿

---

## 总体进度看板

| Issue | 任务 | 状态 | 优先级 |
|-------|------|------|--------|
| 1 | 决策标签生成 | 🔴 待开始 | P0 |
| 2 | Baseline 训练 (E1) | 🔴 待开始 | P0 |
| 3 | Ours 完整模型训练 | 🔴 待开始 | P0 |
| 4 | 规则满足率+物理约束 (E2,E3) | 🔴 待开始 | P1 |
| 5 | 闭环收敛 (E4) | 🔴 待开始 | P1 |
| 6 | 鲁棒性 (E5) | 🔴 待开始 | P1 |
| 7 | 消融实验 (E6) | 🔴 待开始 | P1 |
| 8 | 可解释性案例 (E7) | 🔴 待开始 | P2 |
| 9 | 论文图表生成 | 🔴 待开始 | P1 |
| 10 | 论文 Sec 3: Methodology | 🔴 待开始 | P0 |
| 11 | 论文 Sec 1-2: Intro+Related | 🔴 待开始 | P0 |
| 12 | 论文 Sec 4-5: Exp+Conclusion | 🔴 待开始 | P1 |
| 13 | 论文定稿+投稿 | 🔴 待开始 | P2 |

### 快速启动命令

```bash
# Issue 1: 标签生成
python experiments/label_decision_intents.py

# Issue 2: Baseline (快速验证)
python experiments/exp_e1_decision_accuracy.py --model lstm --epochs 5 --zones-per-split 2

# Issue 3: Ours 完整模型 (快速验证)
python experiments/exp_ours_full.py --epochs 5 --zones-per-split 2

# Issue 2+3 完整运行
bash experiments/run_all_experiments.sh

# Issue 4-8 评估（依赖已训练好的 checkpoint）
python experiments/exp_evaluation.py --exp e2 --checkpoint checkpoints/ours_full_best.pth
python experiments/exp_evaluation.py --exp e3 --checkpoint checkpoints/ours_full_best.pth
python experiments/exp_evaluation.py --exp e4 --checkpoint checkpoints/ours_full_best.pth
python experiments/exp_evaluation.py --exp e5 --checkpoint checkpoints/ours_full_best.pth
python experiments/exp_evaluation.py --exp e7 --checkpoint checkpoints/ours_full_best.pth

# Issue 7: 消融实验
python experiments/exp_e6_ablation.py --epochs 20 --zones-per-split 3

# Issue 9: 图表生成
python experiments/plot_results.py --results-dir outputs --output-dir results/paper_figures/
```
