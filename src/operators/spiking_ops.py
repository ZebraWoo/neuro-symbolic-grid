'''
1. Izhikevich神经元算子
状态变量-膜电位v、恢复变量u
参数-a、b、c、d-控制神经元的动态行为
输入-外部电流I
输出-脉冲序列-0或1
'''
import torch
import torch.nn as nn

class IzhikevichNeuron(nn.Module):
    def __init__(self, a=0.02, b=0.2, c=-65, d=8):
        super(IzhikevichNeuron, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v = None  # 膜电位
        self.u = None  # 恢复变量

    def forward(self, I):
        if self.v is None:
            self.v = torch.full_like(I, self.c)  # 初始化膜电位
            self.u = torch.full_like(I, self.b * self.c)  # 初始化恢复变量

        # 更新膜电位和恢复变量
        dv = (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + I) * 0.5  # 时间步长为0.5ms
        du = (self.a * (self.b * self.v - self.u)) * 0.5

        self.v += dv
        self.u += du

        # 检测是否发生了脉冲
        spikes = (self.v >= 30).float()  # 膜电位达到30mV时产生脉冲
        if spikes.sum() > 0:
            # 重置膜电位和恢复变量
            reset_indices = (self.v >= 30)
            self.v[reset_indices] = self.c
            self.u[reset_indices] += self.d

        return spikes

# LIF神经元算子(前期测试)
class LIFNeuron(nn.Module):
    def __init__(self, tau=20.0, threshold=1.0):
        super(LIFNeuron, self).__init__()
        self.tau = tau
        self.threshold = threshold
        self.v = None  # 膜电位

    def forward(self, I):
        if self.v is None:
            self.v = torch.zeros_like(I)  # 初始化膜电位

        # 更新膜电位
        dv = (-self.v + I) / self.tau
        self.v += dv

        # 检测是否发生了脉冲
        spikes = (self.v >= self.threshold).float()  # 膜电位达到阈值时产生脉冲
        if spikes.sum() > 0:
            # 重置膜电位
            reset_indices = (self.v >= self.threshold)
            self.v[reset_indices] = 0.0

        return spikes


'''
2. 多模态数据编码算子
csv数据(时间、负荷、风电、太阳能等)-> 多模态编码器 -> 融合特征表示
包含内容
- 频率编码 Rate Coding: 将数值数据转换为脉冲频率，数值越大，脉冲频率越高。
- 时间编码 Temporal Coding: 将数值数据转换为脉冲时间，数值越大，脉冲时间越早。
- 位编码 Population Coding: 将数值数据转换为多个神经元的脉冲模式，每个神经元对应一个数值范围。


'''
class MultiModalEncoder(nn.Module):
    def __init__(self, input_size, encoding_size):
        super(MultiModalEncoder, self).__init__()
        self.input_size = input_size
        self.encoding_size = encoding_size
        self.rate_encoder = nn.Linear(input_size, encoding_size)
        self.temporal_encoder = nn.Linear(input_size, encoding_size)
        self.population_encoder = nn.Linear(input_size, encoding_size)

    def forward(self, x):
        # 需要修改
        rate_encoded = torch.relu(self.rate_encoder(x))  # 频率编码
        temporal_encoded = torch.relu(self.temporal_encoder(x))  # 时间编码
        population_encoded = torch.relu(self.population_encoder(x))  # 位编码

        # 融合特征表示，需要探索
        fused_features = rate_encoded + temporal_encoded + population_encoded
        return fused_features


'''
3. 神经符号推理算子
包含内容
- 规则编码 Rule Encoding: 将电力系统的物理规律和约束条件编码为逻辑规则。
- 符号推理 Symbolic Reasoning: 基于编码的规则进行推理，推断系统状态和预测未来行为。
- 解释性 Explanation: 提供推断过程的解释。

'''



'''
4. 伪梯度训练算子
包含内容
- 伪梯度计算 Pseudo-gradient Calculation: 设计适用于脉冲神经网络的伪梯度计算方法，解决非连续性问题。
- 反向传播 Backpropagation: 基于伪梯度进行反向传播，更新模型参数。
- 收敛性 Convergence: 研究伪梯度训练的收敛性和稳定性。

'''
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        # 即使脉冲函数不可导，我们也假装它是一个类似 Sigmoid 的斜坡
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        surrogate_grad = torch.exp(-torch.abs(input)) # 常见的代理梯度示例
        return grad_input * surrogate_grad