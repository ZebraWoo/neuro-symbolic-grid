"""
神经元模型库 - 第①步：主流神经元与突触模型构建技术
包含：LIF、IF、Hodgkin-Huxley等主流神经元模型
以及静态和动态突触模型
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np


class SurrogateSpike(torch.autograd.Function):
    """Heaviside forward + triangular surrogate gradient (same as demo/snn_mlp)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.threshold = threshold
        return (x >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        th = ctx.threshold
        grad_x = grad_output * torch.clamp(1.0 - (x - th).abs(), min=0.0)
        return grad_x, None


class TemporalLIF(nn.Module):
    """
    Batched LIF over a sequence: input (batch, seq_len, dim) -> spikes (batch, seq_len, dim).
    No cross-batch membrane state; safe for DDP and backprop.
    """

    def __init__(self, threshold: float = 1.0, leak: float = 0.9):
        super().__init__()
        self.threshold = threshold
        self.leak = leak

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        mem = torch.zeros(batch_size, dim, device=x.device, dtype=x.dtype)
        spikes = []
        for t in range(seq_len):
            mem = self.leak * mem + x[:, t, :]
            spk = SurrogateSpike.apply(mem, self.threshold)
            mem = mem * (1.0 - spk)
            spikes.append(spk.unsqueeze(1))
        return torch.cat(spikes, dim=1)


class LeakyIntegrateFire(nn.Module):
    """
    泄漏积分发火神经元（LIF）- 最常用的脉冲神经元
    
    方程：
        τ * dV/dt = -V(t) + I(t)
        当 V >= V_th 时发火，重置 V = V_reset
    """
    
    def __init__(
        self,
        input_dim: int,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        v_leak: float = 1.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.v_leak = v_leak
        
        # 突触权重
        self.weight = nn.Parameter(torch.randn(input_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))
        
        # 膜电位
        self.register_buffer('v_mem', torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, input_dim) 输入电流
            
        Returns:
            spike: (batch_size, 1) 脉冲输出 (0或1)
            v_mem: (batch_size, 1) 膜电位
        """
        # 输入电流
        I = torch.matmul(x, self.weight) + self.bias  # (batch_size,)
        
        # 膜电位更新：τ * dV/dt = -V + I
        # 离散形式：V_new = V_old + (1/τ) * (-V_old + I)
        dv = (1.0 / self.tau) * (-self.v_leak * self.v_mem + I)
        self.v_mem = self.v_mem + dv
        
        # 发火：采用替代梯度方法
        spike = self._heaviside_spike(self.v_mem - self.v_threshold)
        
        # 重置
        self.v_mem = self.v_mem * (1 - spike.detach()) + spike.detach() * self.v_reset
        
        return spike, self.v_mem
    
    @staticmethod
    def _heaviside_spike(x: torch.Tensor) -> torch.Tensor:
        """Heaviside阶跃函数 + Sigmoid替代梯度"""
        if x.requires_grad:
            # 前向传播：Heaviside
            spike = (x > 0).float()
            # 梯度：Sigmoid导数（替代梯度法）
            grad = torch.sigmoid(x) * (1 - torch.sigmoid(x))
            return spike + (grad - grad.detach())
        else:
            return (x > 0).float()
    
    def reset_membrane(self, batch_size: int = 1):
        """重置膜电位"""
        self.v_mem = torch.zeros(batch_size, device=self.weight.device)


class IntegrateFire(nn.Module):
    """
    标准积分发火神经元（IF）- LIF的简化版
    
    方程：V(t+1) = V(t) + I(t)
    """
    
    def __init__(
        self,
        input_dim: int,
        v_threshold: float = 1.0,
        v_reset: float = 0.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        
        self.weight = nn.Parameter(torch.randn(input_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))
        
        self.register_buffer('v_mem', torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """简单的一阶积分"""
        I = torch.matmul(x, self.weight) + self.bias
        self.v_mem = self.v_mem + I
        
        spike = (self.v_mem >= self.v_threshold).float()
        self.v_mem = self.v_mem * (1 - spike.detach()) + spike.detach() * self.v_reset
        
        return spike, self.v_mem


class HodgkinHuxley(nn.Module):
    """
    Hodgkin-Huxley模型 - 生物学精准模型，用于复杂行为模拟
    
    方程组：
        C * dV/dt = -g_Na*m³*h*(V-E_Na) - g_K*n⁴*(V-E_K) - g_L*(V-E_L) + I
        + 门控方程：dm/dt = α_m(V)*(1-m) - β_m(V)*m
    
    参数：膜电容C、离子通道电导(g_Na, g_K, g_L)、反向电位
    """
    
    def __init__(
        self,
        input_dim: int,
        C_m: float = 1.0,  # 膜电容
        g_Na: float = 120.0,  # Na通道最大电导
        g_K: float = 36.0,   # K通道最大电导
        g_L: float = 0.3,    # 漏电导
        E_Na: float = 50.0,  # Na反向电位
        E_K: float = -77.0,  # K反向电位
        E_L: float = -54.4,  # 漏电位
        v_rest: float = -65.0  # 静息电位
    ):
        super().__init__()
        self.input_dim = input_dim
        self.C_m = C_m
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_L = g_L
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L
        self.v_rest = v_rest
        
        self.weight = nn.Parameter(torch.randn(input_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))
        
        # 状态变量
        self.register_buffer('v_mem', torch.full((1,), v_rest))
        self.register_buffer('m', torch.zeros(1))  # Na激活
        self.register_buffer('h', torch.zeros(1))  # Na失活
        self.register_buffer('n', torch.zeros(1))  # K激活
        
    def alpha_m(self, v: torch.Tensor) -> torch.Tensor:
        """Na激活门的开启速率"""
        return 0.1 * (v + 40) / (1 - torch.exp(-(v + 40) / 10))
    
    def beta_m(self, v: torch.Tensor) -> torch.Tensor:
        """Na激活门的关闭速率"""
        return 4 * torch.exp(-(v + 65) / 18)
    
    def alpha_h(self, v: torch.Tensor) -> torch.Tensor:
        """Na失活门的开启速率"""
        return 0.07 * torch.exp(-(v + 65) / 20)
    
    def beta_h(self, v: torch.Tensor) -> torch.Tensor:
        """Na失活门的关闭速率"""
        return 1 / (1 + torch.exp(-(v + 35) / 10))
    
    def alpha_n(self, v: torch.Tensor) -> torch.Tensor:
        """K激活门的开启速率"""
        return 0.01 * (v + 55) / (1 - torch.exp(-(v + 55) / 10))
    
    def beta_n(self, v: torch.Tensor) -> torch.Tensor:
        """K激活门的关闭速率"""
        return 0.125 * torch.exp(-(v + 65) / 80)
    
    def forward(self, x: torch.Tensor, dt: float = 0.01) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x: (batch_size, input_dim) 输入电流
            dt: 时间步长
            
        Returns:
            spike: 脉冲 (激发时为1)
            state_dict: 包含V, m, h, n的状态字典
        """
        # 输入电流
        I_ext = torch.matmul(x, self.weight) + self.bias
        
        # 离子电流
        I_Na = self.g_Na * (self.m ** 3) * self.h * (self.v_mem - self.E_Na)
        I_K = self.g_K * (self.n ** 4) * (self.v_mem - self.E_K)
        I_L = self.g_L * (self.v_mem - self.E_L)
        
        # 膜电位更新
        dv = (I_ext - I_Na - I_K - I_L) / self.C_m
        self.v_mem = self.v_mem + dt * dv
        
        # 门控变量更新（欧拉法）
        self.m = self.m + dt * (self.alpha_m(self.v_mem) * (1 - self.m) - 
                                 self.beta_m(self.v_mem) * self.m)
        self.h = self.h + dt * (self.alpha_h(self.v_mem) * (1 - self.h) - 
                                 self.beta_h(self.v_mem) * self.h)
        self.n = self.n + dt * (self.alpha_n(self.v_mem) * (1 - self.n) - 
                                 self.beta_n(self.v_mem) * self.n)
        
        # 钳位到有效范围
        self.m = torch.clamp(self.m, 0, 1)
        self.h = torch.clamp(self.h, 0, 1)
        self.n = torch.clamp(self.n, 0, 1)
        
        # 发火：动作电位阈值约为-40mV
        spike = (self.v_mem > -40).float()
        
        return spike, {
            'v': self.v_mem.detach(),
            'm': self.m.detach(),
            'h': self.h.detach(),
            'n': self.n.detach(),
            'I_Na': I_Na.detach(),
            'I_K': I_K.detach(),
            'I_L': I_L.detach()
        }


# ==================== 突触模型 ====================

class StaticSynapse(nn.Module):
    """
    静态突触模型 - 简单线性权重
    
    方程：s_post = w * s_pre
    """
    
    def __init__(self, pre_neurons: int, post_neurons: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(pre_neurons, post_neurons) * 0.01)
        self.bias = nn.Parameter(torch.zeros(post_neurons))
        
    def forward(self, spikes_pre: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spikes_pre: (batch_size, pre_neurons) 前神经元脉冲
            
        Returns:
            current: (batch_size, post_neurons) 后神经元接收的电流
        """
        return torch.matmul(spikes_pre, self.weight) + self.bias


class DynamicSynapse(nn.Module):
    """
    动态突触模型 - 支持短期易化(STF)和短期抑制(STD)
    
    模型参数：
        U: 释放概率
        tau_f: 易化时间常数
        tau_d: 抑制时间常数
    """
    
    def __init__(
        self,
        pre_neurons: int,
        post_neurons: int,
        U: float = 0.5,  # 初始释放概率
        tau_f: float = 50.0,  # 易化时间常数 (ms)
        tau_d: float = 200.0  # 抑制时间常数 (ms)
    ):
        super().__init__()
        self.pre_neurons = pre_neurons
        self.post_neurons = post_neurons
        self.U = U
        self.tau_f = tau_f
        self.tau_d = tau_d
        
        # 静态权重
        self.weight = nn.Parameter(torch.randn(pre_neurons, post_neurons) * 0.01)
        
        # 动态资源（易化和抑制）
        self.register_buffer('x', torch.ones(post_neurons))  # 可用资源池 (STD)
        self.register_buffer('u', torch.full((post_neurons,), U))  # 释放概率 (STF)
        
    def forward(
        self,
        spikes_pre: torch.Tensor,
        dt: float = 1.0
    ) -> torch.Tensor:
        """
        Args:
            spikes_pre: (batch_size, pre_neurons) 前神经元脉冲
            dt: 时间步长 (ms)
            
        Returns:
            current: (batch_size, post_neurons) 突触电流
        """
        batch_size = spikes_pre.size(0)
        
        # 计算脉冲触发的释放量
        curr_u = self.u  # 当前释放概率
        release = curr_u * self.x  # 实际释放量 = 释放概率 × 可用资源
        
        # 资源更新（抑制恢复）
        # x' = (1-x)/tau_d - U*x（脉冲时消耗）
        recovery_x = (1 - self.x) / self.tau_d
        self.x = self.x + dt * recovery_x
        
        # 释放概率更新（易化）
        # u' = (U-u)/tau_f + U*(1-u)（脉冲时增加）
        recovery_u = (self.U - self.u) / self.tau_f
        self.u = self.u + dt * recovery_u
        
        # 约束
        self.x = torch.clamp(self.x, 0, 1)
        self.u = torch.clamp(self.u, 0, 1)
        
        # 突触电流：考虑动态强度调制
        base_current = torch.matmul(spikes_pre, self.weight)  # (batch_size, post_neurons)
        modulated_current = base_current * release.unsqueeze(0)  # 动态调制
        
        # 脉冲时消耗资源
        self.x = self.x - release * spikes_pre.sum(dim=0) / (batch_size + 1e-8)
        self.x = torch.clamp(self.x, 0, 1)
        
        return modulated_current


class SynapticPlasticity(nn.Module):
    """
    突触可塑性 - STDP (Spike-Timing Dependent Plasticity)
    
    学习规则：
        Δw ∝ exp(-(t_post - t_pre) / τ)
        - 若前神经元先发火：w增加（长期增强LTP）
        - 若后神经元先发火：w减少（长期抑制LTD）
    """
    
    def __init__(
        self,
        pre_neurons: int,
        post_neurons: int,
        tau_stdp: float = 20.0,  # STDP时间窗口
        A_plus: float = 0.01,   # LTP强度
        A_minus: float = 0.01   # LTD强度
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(pre_neurons, post_neurons) * 0.01)
        self.tau_stdp = tau_stdp
        self.A_plus = A_plus
        self.A_minus = A_minus
        
        # 脉冲历史（用于STDP计算）
        self.register_buffer('spike_time_pre', torch.zeros(pre_neurons))
        self.register_buffer('spike_time_post', torch.zeros(post_neurons))
        
    def forward(
        self,
        spikes_pre: torch.Tensor,
        spikes_post: torch.Tensor,
        t: int = 0
    ) -> torch.Tensor:
        """
        Args:
            spikes_pre: (batch_size, pre_neurons)
            spikes_post: (batch_size, post_neurons)
            t: 当前时刻
            
        Returns:
            current: (batch_size, post_neurons)
        """
        # STDP学习
        for i in range(self.weight.size(0)):
            for j in range(self.weight.size(1)):
                # 时间差
                dt = self.spike_time_post[j] - self.spike_time_pre[i]
                
                # STDP曲线
                if dt > 0:  # 后发火：LTD
                    dw = -self.A_minus * torch.exp(-dt / self.tau_stdp)
                else:  # 前发火：LTP
                    dw = self.A_plus * torch.exp(dt / self.tau_stdp)
                
                self.weight.data[i, j] = self.weight.data[i, j] + dw
        
        # 记录脉冲时刻
        if spikes_pre.sum() > 0:
            self.spike_time_pre = torch.full_like(self.spike_time_pre, t, dtype=self.spike_time_pre.dtype)
        if spikes_post.sum() > 0:
            self.spike_time_post = torch.full_like(self.spike_time_post, t, dtype=self.spike_time_post.dtype)
        
        # 计算电流
        return torch.matmul(spikes_pre, self.weight)


# ==================== 神经元库 ====================

NEURON_MODELS = {
    'LIF': LeakyIntegrateFire,
    'IF': IntegrateFire,
    'HH': HodgkinHuxley,
}

SYNAPSE_MODELS = {
    'static': StaticSynapse,
    'dynamic': DynamicSynapse,
    'stdp': SynapticPlasticity,
}
