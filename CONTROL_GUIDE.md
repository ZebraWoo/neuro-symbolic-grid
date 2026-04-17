# 电网自主调控系统 - 完整技术文档

## 📋 项目目标

构建一个基于脉冲神经网络的**电网自主调控预训练模型系统**，具体目标：

①. 研究主流神经元模型与突触模型构建技术  
②. 设计多种模态下的深度神经网络模型，建立适用于电网调控的新型自主调控预训练模型  
③. 构建Izhikevich和多室模型，支持复杂神经元行为模拟（留接口暂不实现）

---

## 🧠 第①步：神经元与突触模型库

### 文件位置
```
src/control/neuron_models.py  (12.3 KB)
```

### 实现的神经元模型

#### 1. **LIF (Leaky Integrate-Fire)** - 泄漏积分发火
最常用的脉冲神经元模型，计算高效，生物学意义强。

```
数学方程：
  τ * dV/dt = -V(t) + I(t)
  当 V ≥ V_th 时发火，重置 V = V_reset

特点：
  ✓ 一阶线性动力学
  ✓ 替代梯度法处理非可导性
  ✓ 适合大规模网络
```

**使用示例：**
```python
from src.control.neuron_models import LeakyIntegrateFire
neuron = LeakyIntegrateFire(
    input_dim=66,  # 66个地区的输入
    tau=2.0,       # 时间常数
    v_threshold=1.0
)
spike, v_mem = neuron(input_current)  # 输出脉冲和膜电位
```

#### 2. **IF (Integrate-Fire)** - 标准积分发火
LIF的简化版，无漏电项。

```
数学方程：
  V(t+1) = V(t) + I(t)
  
特点：
  ✓ 计算最简单
  ✓ 适合轻量级应用
  ✓ 可作为baseline对比
```

#### 3. **Hodgkin-Huxley** - 生物学精确模型
基于真实神经元的电生理学，支持复杂动作电位行为。

```
方程组：
  C_m * dV/dt = -g_Na*m³*h*(V-E_Na) - g_K*n⁴*(V-E_K) - g_L*(V-E_L) + I
  
  门控变量动力学：
    dm/dt = α_m(V)*(1-m) - β_m(V)*m
    dh/dt = α_h(V)*(1-h) - β_h(V)*h
    dn/dt = α_n(V)*(1-n) - β_n(V)*n

参数含义：
  - C_m: 膜电容
  - g_Na, g_K, g_L: 离子通道最大电导
  - E_Na, E_K, E_L: 反向电位
  - m, h, n: 门控变量（激活/失活概率）

特点：
  ✓ 最生物学精确
  ✓ 支持不同激发模式（阈下振荡、重复发火等）
  ✓ 计算开销大，适合少数关键神经元
```

### 突触模型

#### 1. **StaticSynapse** - 静态突触
简单的权重乘积，突触强度固定不变。

```
方程：I_post = w * s_pre

特点：
  ✓ 最简单，计算快
  ✓ 无动态行为
  ✓ baseline模型
```

#### 2. **DynamicSynapse** - 动态突触
支持**短期易化(STF)**和**短期抑制(STD)**，模拟真实神经元的适应性。

```
关键机制：
  - 短期抑制(STD): 重复刺激导致突触强度减弱
    x' = (1-x)/τ_d - U*x (x为可用资源)
    
  - 短期易化(STF): 前期脉冲增加释放概率
    u' = (U-u)/τ_f + U*(1-u) (u为释放概率)
    
  - 实际释放量 = u * x
    
参数：
  - U: 初始释放概率
  - τ_f: 易化时间常数(~50ms)
  - τ_d: 抑制时间常数(~200ms)

应用场景：
  ✓ 模拟高频刺激下的神经元疲劳
  ✓ 突触信息过滤
  ✓ 时间相关学习
```

#### 3. **SynapticPlasticity** - STDP学习
**Spike-Timing Dependent Plasticity** - 基于脉冲时序的突触可塑性。

```
学习规则：
  Δw ∝ exp(-(t_post - t_pre) / τ)
  
  - 若前神经元先发火(t_pre < t_post)：LTP (长期增强) → w↑
  - 若后神经元先发火(t_pre > t_post)：LTD (长期抑制) → w↓
  
时间窗口：
  - τ_STDP ≈ 20ms (脉冲时序影响范围)
  - 超出此范围，学习效应消失

应用：
  ✓ 无监督学习规则
  ✓ 因果关系学习
  ✓ 特征提取
```

---

## 🎯 第②步：多模态自主调控预训练模型

### 文件位置
```
src/control/multimodal_control_network.py  (14.5 KB)
```

### 核心架构

```
输入数据 (多模态)
  ↓
[多模态嵌入层] ← 融合各模态到共同特征空间
  ├─ 负荷数据编码器 (66个地区, 720时步)
  ├─ 电压数据编码器 (66个地区, 动态特征)
  ├─ 频率数据编码器 (系统频率, 1维)
  ├─ 天气数据编码器 (温度、风速、光照, 3维)
  ├─ 时间特征编码器 (小时、日、周、季, 4维)
  └─ [融合门] ← 学习每个模态的权重
  ↓
[脉冲Transformer块] × 4  ← 时间序列学习和因果关系建模
  ├─ 自注意力层 (捕捉长程依赖)
  ├─ 前馈网络 (特征变换)
  └─ LIF神经元层 (脉冲决策)
  ↓
[调控决策头] ← 生成具体控制指令
  ├─ 频率控制量
  ├─ 电压控制量
  ├─ 负荷控制量
  ├─ 应急响应
  └─ 预留控制
  ↓
模型输出
  ├─ control_actions: (batch_size, 5) - 调控指令
  ├─ confidence: (batch_size,) - 决策置信度
  ├─ modality_weights: 模态重要性权重
  └─ spike_rates: 神经元发火率统计
```

### 多模态融合

**问题**：不同来源的数据有不同的物理含义和量纲。

**解决方案**：为每个模态设计独立编码器，通过融合门学习权重。

```python
# 使用示例
modalities_data = {
    'load': torch.randn(32, 720, 66),      # 负荷 (batch, time, zones)
    'voltage': torch.randn(32, 720, 66),   # 电压
    'frequency': torch.randn(32, 720, 1),  # 频率
    'weather': torch.randn(32, 720, 3),    # 天气
    'time': torch.randn(32, 720, 4),       # 时间特征
}

model = MultimodalControlNetwork(
    modalities={
        'load': 66, 'voltage': 66, 'frequency': 1,
        'weather': 3, 'time': 4
    },
    embedding_dim=32,
    seq_len=720
)

output = model(modalities_data)
# 输出：
#   - control_actions: (32, 5) 调控指令
#   - confidence: (32,) 置信度
#   - modality_weights: (32, 5) 模态权重
```

### 脉冲Transformer块

结合自注意力和脉冲神经元的混合架构。

```
输入特征 (batch, seq_len, embedding_dim)
  ↓
[多头自注意力] + LayerNorm
  ↓
[前馈网络] + LayerNorm
  ↓
[LIF神经元层] (32个神经元)
  ↓ 输出脉冲和发火率
[权重融合] 
  ↓
最终输出 = 注意力输出 + 0.3 * 脉冲输出
```

**优势**：
- 自注意力捕捉全局依赖
- LIF神经元进行脉冲决策，学习稀疏表示
- 混合设计结合两者优点

### 四组分预训练损失函数

```python
总损失 = w_recon * L_recon 
       + w_contrast * L_contrast 
       + w_consist * L_consistency 
       + w_uncert * L_uncertainty

其中：
```

#### 1. **重建损失** (0.3权重)
```
L_recon = MSE(原始数据, 重建数据)

目的：模型需要理解各模态的内在结构
```

#### 2. **对比损失** (0.3权重)
```
L_contrast = Σ_t (1 - cosine_sim(emb_t, emb_{t+1}))

目的：相邻时刻的嵌入表示应相似
      学习时间连贯性
```

#### 3. **调控一致性损失** (0.2权重)
```
L_consistency = Σ_t ||control_t - control_{t+1}||²

目的：防止调控指令频繁振荡
      学习决策的稳定性
```

#### 4. **不确定性正则化** (0.2权重)
```
L_uncertainty = -mean(log(confidence + 1e-6))

目的：鼓励模型对高置信度决策
      自适应调整决策风险
```

---

## 🔬 第③步：高级神经元模型接口

### 文件位置
```
src/control/advanced_neuron_models.py  (11.9 KB)
```

### Izhikevich神经元接口

两变量的简化Izhikevich模型，支持丰富的发火行为。

```
数学方程：
  dV/dt = 0.04*V² + 5*V + 140 - U + I
  dU/dt = a*(b*V - U)
  
  当 V ≥ 30 时发火，重置：V ← c, U ← U + d

参数解释：
  - V: 膜电位 (mV)
  - U: 恢复电流 (调节重极化)
  - a: 恢复时间标度
  - b: 恢复灵敏度
  - c: 发火后重置电位
  - d: 回后超极化
```

#### 神经元类型参数化

```python
from src.control.advanced_neuron_models import IzhikevichNeuronInterface

# 规则尖峰 (Regular Spiking, RS) - 最常见
neuron_rs = IzhikevichNeuronInterface(
    input_dim=66,
    neuron_type='RS',
    a=0.02, b=0.2, c=-65, d=8
)

# 快速尖峰 (Fast Spiking, FS) - 抑制性神经元
neuron_fs = IzhikevichNeuronInterface(
    input_dim=66,
    neuron_type='FS',
    a=0.1, b=0.2, c=-65, d=2
)

# 低阈值尖峰 (Low Threshold Spiking, LTS)
neuron_lts = IzhikevichNeuronInterface(
    input_dim=66,
    neuron_type='LTS',
    a=0.02, b=0.25, c=-65, d=2
)

# 内在爆炸 (Intrinsically Bursting, IB) - 产生突发脉冲
neuron_ib = IzhikevichNeuronInterface(
    input_dim=66,
    neuron_type='IB',
    a=0.02, b=0.2, c=-55, d=4
)
```

**发火行为对比**：
| 类型 | 特征 | 应用场景 |
|------|------|--------|
| RS | 规则单脉冲 | 主要的兴奋性输出 |
| FS | 高频持续发火 | 快速信息处理 |
| LTS | 低阈值、低频 | 阈值检测 |
| IB | 突发脉冲簇 | 高强度刺激响应 |

### 多室神经元模型

支持分布式神经元，不同室有独立的膜电位和电流。

```
架构：
  树突 (Dendrite)  ← 接收突触输入
    ↓ (轴向耦合)
  胞体 (Soma)      ← 决策和发火
    ↓ (轴向耦合)
  轴突 (Axon)      ← 产生脉冲输出

方程：
  C_m * dV_i/dt = -g_L*(V_i - E_L) 
                  + Σ I_synaptic 
                  + Σ I_coupling  
                  + I_ext
  
  I_coupling = g_coupling * (V_j - V_i)  (室间耦合)

特点：
  ✓ 生物学精度高
  ✓ 支持复杂的时空动力学
  ✓ 计算开销较大 (~10倍vs LIF)
```

**使用示例**：
```python
from src.control.advanced_neuron_models import MultiCompartmentNeuronInterface

neuron = MultiCompartmentNeuronInterface(
    input_dim=66,
    num_compartments=3,  # soma, dendrite, axon
    C_m=1.0,
    g_L=0.1,
    E_L=-70.0,
    g_coupling=0.5  # 室间耦合强度
)

spike, state = neuron(input_current)
# state['V']: (3, batch_size) - 各室膜电位
# state['I_coupling']: (3, batch_size) - 室间耦合电流
```

### 混合神经元网络

在同一网络中混用多种神经元模型，灵活选择计算精度和速度。

```python
from src.control.advanced_neuron_models import HybridNeuronNetwork

config = [
    {
        'type': 'LIF',
        'input_dim': 66,
        'output_dim': 32,
        'tau': 2.0
    },
    {
        'type': 'IZH',
        'input_dim': 32,
        'output_dim': 16,
        'neuron_type': 'RS'
    },
    {
        'type': 'MULTI',
        'input_dim': 16,
        'output_dim': 8,
        'num_compartments': 3
    }
]

network = HybridNeuronNetwork(config, input_dim=66)
output = network(input_data)
```

**分层设计建议**：
1. 输入层：LIF (高效处理高维输入)
2. 中间层：Izhikevich (平衡精度和速度)
3. 输出层：MultiCompartment (精确决策)

---

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install torch torchvision torchaudio
```

### 2. 运行训练
```bash
# 多模态模型训练
bash train_control.sh multimodal 50 32 0.001 cuda

# 参数说明：
#   $1: 模型类型 (multimodal|lif|izh|hybrid)
#   $2: 训练轮数 (epochs)
#   $3: 批次大小 (batch_size)
#   $4: 学习率 (lr)
#   $5: 计算设备 (cuda|cpu)
```

### 3. 检查输出
```bash
ls -la checkpoints/control_model.pth
cat outputs/training_history.json
```

---

## 📊 项目结构

```
electric-power-regulation/
├── src/control/
│   ├── __init__.py
│   ├── neuron_models.py              # ①神经元和突触库
│   ├── multimodal_control_network.py # ②多模态调控网络
│   └── advanced_neuron_models.py     # ③高级模型接口
├── train_control.sh                  # 训练脚本
├── checkpoints/
│   └── control_model.pth             # 保存的模型权重
├── outputs/
│   └── training_history.json         # 训练历史
└── CONTROL_GUIDE.md                  # 本文档
```

---

## 📈 性能对比

| 指标 | LIF | Izhikevich | HH | Multi-Compartment |
|------|-----|-----------|----|--------------------|
| 计算速度 | ⚡⚡⚡ | ⚡⚡ | ⚡ | 🔌 |
| 生物学精度 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 模型复杂度 | 低 | 中 | 高 | 高 |
| 发火行为 | 简单 | 丰富 | 最完整 | 最完整 |
| 适用场景 | 大规模网络 | 平衡应用 | 关键神经元 | 精确模拟 |

---

## 🔧 常见问题

### Q: 如何选择神经元模型？
**A**: 
- **快速原型**：使用LIF (最快)
- **平衡应用**：使用Izhikevich (推荐)
- **精确研究**：使用HH或MultiCompartment (精度最高)
- **生产系统**：使用混合策略 (输入层LIF，关键层Izhikevich)

### Q: 多模态融合有什么优势？
**A**:
1. 信息互补：不同模态捕捉不同方面
2. 鲁棒性强：单模态缺失时仍可工作
3. 可解释性：融合权重显示各模态重要性
4. 泛化能力：多源数据学习更好的表示

### Q: 预训练损失中为什么要加入不确定性项？
**A**:
1. 防止过度自信导致错误决策
2. 自适应调整决策风险
3. 在高不确定性场景下降低控制强度
4. 与贝叶斯解释一致

---

## 📚 参考文献

1. Izhikevich EM. (2004). "Which model to use for cortical spiking neurons?"
2. Brette R, Gerstner W. (2005). "Adaptive exponential integrate-and-fire model"
3. Hodgkin AL, Huxley AF. (1952). "A quantitative description of membrane current"
4. Tsodyks M, Pawelzik K, Markram H. (1998). "Neural Networks with Dynamic Synapses"
5. Abbott LF. (1999). "Lapique's introduction of the integrate-and-fire model neuron (1907)"

---

## 👥 贡献和反馈

如有建议或发现问题，请联系项目团队。

**更新日期**：2026-04-17  
**版本**：1.0 - 核心框架完成
