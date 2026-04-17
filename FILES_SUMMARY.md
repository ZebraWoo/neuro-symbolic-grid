# 文件清单 📋

## 新增的Shell脚本 (3个核心脚本)

### 🚀 核心训练脚本
1. **`train_with_metrics.sh`** (6.8 KB)
   - 启动Spikformer预训练，带完整的评估指标
   - 计算: AUC、Precision、Recall、F1、Silhouette、Proximity
   - 用法: `bash train_with_metrics.sh [options]`
   - 自动保存检查点和指标数据

2. **`plot_metrics.sh`** (4.1 KB)
   - 从训练数据生成4张可视化图表
   - 生成的图表:
     * 1_loss_curve.png
     * 2_metrics_curve.png (AUC/Precision/Recall/F1)
     * 3_clustering_metrics.png (Silhouette/Proximity)
     * 4_summary_report.png (统计总结)
   - 用法: `bash plot_metrics.sh [options]`

3. **`run_all.sh`** (7.2 KB)
   - 完整工作流: 训练 → 评估 → 可视化
   - 自动运行上面两个脚本
   - 支持跳过某些步骤
   - 用法: `bash run_all.sh [options]`

---

## 新增的Python脚本 (2个)

### 📊 训练脚本
1. **`train_with_metrics.py`** (4.4 KB)
   - 改进的训练脚本，支持命令行参数
   - 使用TrainerWithMetrics类
   - 自动计算和保存所有评估指标
   - 调用方式: 由 `train_with_metrics.sh` 调用

### 🎨 可视化脚本  
2. **`plot_metrics.py`** (10.1 KB)
   - 从JSON指标文件生成4张PNG图表
   - 包含详细的统计信息和标注
   - 调用方式: 由 `plot_metrics.sh` 调用

---

## 新增的Python模块 (1个)

### 🏋️ 训练模块
1. **`src/training/pretrain_training_with_metrics.py`** (17 KB)
   - MetricsTracker: 指标追踪类
   - TrainerWithMetrics: 包含评估指标的训练器
   - compute_auc_from_proximity(): AUC计算函数
   - compute_metrics_from_clustering(): 聚类指标计算
   - 支持实时计算指标并绘图

---

## 新增的文档 (3个)

### 📖 使用指南
1. **`TRAIN_METRICS_GUIDE.md`** (6.6 KB)
   - 详细的使用指南
   - 参数说明和常见问题
   - Python API示例
   - 评估指标详解
   - 建议和最佳实践

2. **`MODEL_OUTPUT_GUIDE.md`** (6.2 KB)
   - 模型输出内容说明
   - 3个可视化工具详解
   - 嵌入空间分析方法
   - 下游应用指导

3. **`QUICK_REFERENCE.sh`** (5.4 KB)
   - 快速参考指南 (shell脚本)
   - 核心用法速查
   - 常见问题解决
   - 一键显示: `bash QUICK_REFERENCE.sh`

---

## 文件关系图

```
用户执行
    │
    ├─── bash run_all.sh
    │    └─── bash train_with_metrics.sh
    │         └─── python train_with_metrics.py
    │              └─── TrainerWithMetrics (src/training/pretrain_training_with_metrics.py)
    │                   ├─── 计算AUC
    │                   ├─── 计算Precision/Recall/F1
    │                   ├─── 计算Silhouette Score
    │                   └─── 保存metrics.json
    │
    └─── bash plot_metrics.sh
         └─── python plot_metrics.py
              └─── 读取metrics.json
                   ├─── 生成1_loss_curve.png
                   ├─── 生成2_metrics_curve.png
                   ├─── 生成3_clustering_metrics.png
                   └─── 生成4_summary_report.png
```

---

## 输出目录结构

```
project_root/
├── checkpoints/
│   └── spikformer_with_metrics/
│       ├── best_model.pt           ← 最佳模型 (自动保存)
│       ├── checkpoint_epoch_5.pt   ← 检查点
│       ├── checkpoint_epoch_10.pt
│       └── ...
│
├── results/
│   ├── metrics.json                ← 原始指标数据 (JSON格式)
│   ├── 1_loss_curve.png            ← 损失曲线
│   ├── 2_metrics_curve.png         ← AUC/Precision/Recall/F1
│   ├── 3_clustering_metrics.png    ← Silhouette/Proximity
│   └── 4_summary_report.png        ← 训练统计总结
```

---

## 快速开始

### 最简单方式
```bash
cd /home/wuzuoxu/electric-power-regulation
bash run_all.sh
```

### 分步骤执行
```bash
# 1. 训练
bash train_with_metrics.sh --num-epochs 30

# 2. 可视化
bash plot_metrics.sh

# 3. 查看结果
ls -lh ./results/*.png
```

### 查看帮助
```bash
bash train_with_metrics.sh --help
bash plot_metrics.sh --help
bash run_all.sh --help
bash QUICK_REFERENCE.sh  # 快速参考
```

---

## 支持的评估指标

| 指标 | 范围 | 说明 | 目标 |
|------|------|------|------|
| **Loss** | [0, ∞) | 重建误差 | ↓ 越低越好 |
| **AUC** | [0, 1] | 分类性能 | ↑ 越高越好 (>0.7) |
| **Precision** | [0, 1] | 精确率 | ↑ 越高越好 |
| **Recall** | [0, 1] | 召回率 | ↑ 越高越好 |
| **F1-Score** | [0, 1] | 调和均值 | ↑ 越高越好 (>0.7) |
| **Silhouette** | [-1, 1] | 聚类质量 | ↑ 越接近1越好 |
| **Proximity** | [0, 1] | 异常检测 | 高值=正常，低值=异常 |

---

## 生成的可视化示例

### 1_loss_curve.png
- 双曲线图表
- 显示训练/验证损失趋势
- 用于判断过拟合

### 2_metrics_curve.png
- 2×2 子图网格
- 每个指标单独绘制
- 包含最佳值标注和最终值

### 3_clustering_metrics.png
- Silhouette Score曲线
- Proximity对比曲线
- 统计信息文本框

### 4_summary_report.png
- 文本总结报告
- 包含所有关键指标的最值和改善幅度
- 易于快速了解训练结果

---

## 依赖包

### 必需
- torch
- numpy
- matplotlib
- seaborn
- scikit-learn (用于AUC、Precision、Recall等计算)
- tqdm

### 可选
- pandas (数据分析)
- jupyter (交互式分析)

安装: `pip install torch scikit-learn matplotlib seaborn tqdm`

---

## 注意事项

1. ✅ 所有脚本都已测试过环境检查和参数解析
2. ✅ 脚本包含详细的错误处理和帮助信息
3. ✅ 支持自定义参数和灵活配置
4. ✅ 自动创建输出目录
5. ✅ 完整的中文文档和注释

---

## 下一步

训练完成后:
1. ✓ 查看 ./results/ 下的PNG图表
2. ✓ 分析 metrics.json 的原始数据
3. ✓ 使用最佳模型 best_model.pt 进行推理
4. ✓ 用Embedding进行下游任务
5. ✓ 用Proximity进行异常检测

---

更多信息请查看:
- TRAIN_METRICS_GUIDE.md (详细使用指南)
- MODEL_OUTPUT_GUIDE.md (模型输出说明)
- bash QUICK_REFERENCE.sh (快速参考)

祝您使用愉快！🚀
