# 论文引用位置对照表

## I. Introduction

**Para 1 — 新能源电网背景**
```
...renewable generation exhibits strong stochasticity, multi-timescale
volatility, and weather-dependent uncertainty [29].
```
→ [29] Hong et al., "Energy Forecasting: A Review and Outlook," 2020.

**Para 2 — 现有方法三分类**
```
Physics-based optimization methods, such as OPF and MPC [22, 23]...
```
→ [22] Fioretto et al., "Predicting AC optimal power flows," AAAI 2020.
→ [23] Chatzos et al., "Spatial network decomposition for fast AC-OPF," IEEE TPWRS 2022.

```
Deep learning methods, including RNNs, TCNs, and Transformer architectures [37, 38, 39]...
```
→ [37] Hochreiter & Schmidhuber, "LSTM," Neural Computation 1997.
→ [38] Vaswani et al., "Attention is all you need," NeurIPS 2017.
→ [39] Bai et al., "An empirical evaluation of TCN," arXiv 2018.

```
...black-box function approximators...offer limited interpretability [31, 33].
```
→ [31] Wang et al., "Review of smart meter data analytics," IEEE TSG 2019.
→ [33] Donti & Kolter, "ML for Sustainable Energy Systems," 2021.

```
PINNs partially address the physical compliance gap [19, 20, 21]...
```
→ [19] Raissi et al., "PINNs," JCP 2019.
→ [20] Karniadakis et al., "Physics-informed ML," Nature Reviews Physics 2021.
→ [21] Misyris et al., "PINNs for power systems," PESGM 2020.

**Para 3 — SNN + Neuro-symbolic**
```
SNNs...governed by biologically plausible neuronal dynamics such as
LIF or Izhikevich models [1, 2].
```
→ [1] Maass, "Networks of spiking neurons," Neural Networks 1997.
→ [2] Izhikevich, "Simple model of spiking neurons," IEEE TNN 2002.

```
...event-driven computation offers inherent advantages in energy
efficiency [5, 10].
```
→ [5] Roy et al., "Towards spike-based machine intelligence," Nature 2019.
→ [10] Eshraghian et al., "Training SNNs using lessons from DL," Proc. IEEE 2023.

```
...surrogate gradient learning [3, 4]...
```
→ [3] Wu et al., "Spatio-temporal backpropagation for SNNs," Frontiers Neurosci. 2018.
→ [4] Neftci et al., "Surrogate gradient learning in SNNs," IEEE Signal Processing 2019.

```
Neuro-symbolic learning aims to combine neural networks with
symbolic reasoning [11, 15].
```
→ [11] d'Avila Garcez et al., "Neural-symbolic computing," arXiv 2019.
→ [15] De Raedt et al., "From statistical relational to neuro-symbolic AI," IJCAI 2020.

**Para 4 — 贡献段 + 转折句**
```
...differentiable soft truth value → semantic loss [12].
```
→ [12] Xu et al., "A semantic loss function for DL with symbolic knowledge," ICML 2018.

```
...temperature-scaled sigmoid logic neuron → Logic Tensor Networks [13, 18].
```
→ [13] Serafini & d'Avila Garcez, "Logic Tensor Networks," CEUR 2016.
→ [18] Badreddine et al., "Logic Tensor Networks," Artificial Intelligence 2022.

## II. Related Work

### A. Data-driven Power Grid Intelligence

```
...data-driven approaches in power grid intelligence [27, 36].
```
→ [27] Li & Du, "Deep Learning for Power System Applications," Springer 2023.
→ [36] Saffari & Khodayar, "Spatiotemporal DL for Power Systems," IEEE Access 2024.

```
Traditional statistical methods (ARIMA, Kalman filtering) [29]...
```
→ [29] Hong et al., "Energy Forecasting," 2020.

```
Deep learning architectures (CNN, LSTM, GRU, Transformer, Informer) [31, 32].
```
→ [31] Wang et al., "Review of smart meter data analytics," IEEE TSG 2019.
→ [32] Wang et al., "Deep learning for probabilistic wind power forecasting," Applied Energy 2017.

```
Graph learning frameworks (GNN, STGCN) [35].
```
→ [35] Dobbe et al., "Toward distributed energy services," IEEE TSG 2020.

```
Reinforcement learning approaches [28, 30].
```
→ [28] Glavic et al., "RL for electric power system decision," IFAC 2017.
→ [30] Vázquez-Canteli & Nagy, "RL for demand response," Applied Energy 2019.

### B. Spiking Neural Networks

```
SNNs as third generation of neural networks [1, 6].
```
→ [1] Maass, "Networks of spiking neurons," 1997.
→ [6] Tavanaei et al., "Deep learning in SNNs," Neural Networks 2019.

```
LIF and Izhikevich neuron models [2, 8].
```
→ [2] Izhikevich, "Simple model of spiking neurons," 2002.
→ [8] Fang et al., "Learnable membrane time constant," ICCV 2021.

```
Surrogate gradient and backpropagation [3, 4, 10].
```
→ [3] Wu et al., "Spatio-temporal backpropagation," 2018.
→ [4] Neftci et al., "Surrogate gradient learning," 2019.
→ [10] Eshraghian et al., "Training SNNs," Proc. IEEE 2023.

```
Spike-timing-dependent plasticity [9].
```
→ [9] Diehl & Cook, "Unsupervised learning with STDP," 2015.

```
Neuromorphic hardware and energy efficiency [5].
```
→ [5] Roy et al., "Towards spike-based machine intelligence," Nature 2019.

```
Spikformer: SNN + Transformer [7].
```
→ [7] Zhou et al., "Spikformer," ICLR 2023.

### C. Physics-informed and Neuro-symbolic Methods

```
PINNs and physics-informed ML [19, 20, 24].
```
→ [19] Raissi et al., "PINNs," JCP 2019.
→ [20] Karniadakis et al., "Physics-informed ML," 2021.
→ [24] Wang et al., "Gradient flow pathologies in PINNs," SIAM 2021.

```
Physics-informed power systems [21, 25, 26].
```
→ [21] Misyris et al., "PINNs for power systems," PESGM 2020.
→ [25] Zamzam & Sidiropoulos, "Physics-Aware NNs for Distribution Systems," IEEE TPWRS 2020.
→ [26] Zamzam & Baker, "Learning optimal solutions for AC-OPF," SmartGridComm 2020.

```
Neuro-symbolic learning foundations [11, 13, 18].
```
→ [11] d'Avila Garcez et al., "Neural-symbolic computing," 2019.
→ [13] Serafini & d'Avila Garcez, "Logic Tensor Networks," 2016.
→ [18] Badreddine et al., "Logic Tensor Networks," AIJ 2022.

```
Semantic loss and differentiable logic [12, 16, 17].
```
→ [12] Xu et al., "Semantic loss function," ICML 2018.
→ [16] Hu et al., "Harnessing DNNs with logic rules," ACL 2016.
→ [17] Rocktäschel & Riedel, "End-to-end differentiable proving," NeurIPS 2017.

```
Neural probabilistic logic [14].
```
→ [14] Manhaeve et al., "DeepProbLog," NeurIPS 2018.

## III. Proposed Framework

### B. Spike Encoding
```
Rate coding strategies [1, 6].
```
→ [1], [6]

### C. SNN Backbone
```
LIF dynamics and membrane time constant [2, 8].
```
→ [2], [8]

```
Surrogate gradient for spike function [3, 4].
```
→ [3], [4]

### D. Neuro-symbolic Rule Layer
```
Differentiable t-norm and soft logic [13, 18].
```
→ [13], [18]

```
Semantic loss regularization [12].
```
→ [12]

### E. Physics-constrained Learning
```
PINN-style constraint encoding [19, 20].
```
→ [19], [20]

```
Power system physical constraints [21, 25].
```
→ [21], [25]

### F. Closed-loop Refinement
```
Self-consistency and unsupervised refinement [11, 15].
```
→ [11], [15]

## IV. Experiments

```
Adam optimizer [40].
```
→ [40] Kingma & Ba, "Adam," ICLR 2015.

```
LSTM baseline [37].
```
→ [37] Hochreiter & Schmidhuber, "LSTM," 1997.

```
Transformer baseline [38].
```
→ [38] Vaswani et al., "Attention is all you need," NeurIPS 2017.

```
TCN baseline [39].
```
→ [39] Bai et al., "An empirical evaluation of TCN," 2018.

```
Izhikevich neuron comparison [2].
```
→ [2] Izhikevich, 2002.

## 引用频次统计

| 引用 | 次数 | 关键段落 |
|------|------|---------|
| [1] Maass 1997 | 3 | Intro + Related B + III.B |
| [2] Izhikevich 2002 | 4 | Intro + Related B + III.C + IV.C |
| [3] Wu 2018 | 3 | Intro + Related B + III.C |
| [4] Neftci 2019 | 3 | Intro + Related B + III.C |
| [5] Roy 2019 | 2 | Intro + Related B |
| [10] Eshraghian 2023 | 2 | Intro + Related B |
| [11] d'Avila Garcez 2019 | 3 | Intro + Related C + III.F |
| [12] Xu 2018 | 3 | Intro + Related C + III.D |
| [13] Serafini 2016 | 3 | Intro + Related C + III.D |
| [15] De Raedt 2020 | 2 | Intro + III.F |
| [18] Badreddine 2022 | 3 | Intro + Related C + III.D |
| [19] Raissi 2019 | 3 | Intro + Related C + III.E |
| [20] Karniadakis 2021 | 3 | Intro + Related C + III.E |
| [21] Misyris 2020 | 3 | Intro + Related C + III.E |
| [37] Hochreiter 1997 | 3 | Intro + Related A + IV.C |
| [38] Vaswani 2017 | 3 | Intro + Related A + IV.C |
| [39] Bai 2018 | 3 | Intro + Related A + IV.C |
| [40] Kingma 2015 | 1 | IV.A |

其余 [6-9, 14, 16, 17, 22-36] 各 1-2 次，分布在 Related Work 三节中。
