"""
Izhikevich模型与多室模型接口 - 第③步
为复杂神经元行为模拟留接口，支持未来扩展
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from abc import ABC, abstractmethod


class NeuronInterface(ABC):
    """
    神经元模型的抽象接口 - 所有具体实现应继承此类
    
    定义神经元模型的标准接口，确保模块化设计和可互换性
    """
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        神经元前向传播
        
        Args:
            x: 输入电流或电压
            
        Returns:
            spike: 脉冲输出
            state: 神经元状态字典
        """
        pass
    
    @abstractmethod
    def reset(self):
        """重置神经元内部状态"""
        pass


class IzhikevichNeuronInterface(nn.Module, NeuronInterface):
    """
    Izhikevich神经元模型接口
    
    完整的两变量Izhikevich模型：
        dV/dt = 0.04*V² + 5*V + 140 - U + I
        dU/dt = a*(b*V - U)
        
    当V ≥ 30时发火，重置V=c, U=U+d
    
    注意：这是接口定义，具体参数化版本可根据需求实现
    支持多种神经元类型参数化：
        - 规则尖峰（Regular Spiking, RS）
        - 快速尖峰（Fast Spiking, FS）
        - 低阈值尖峰（Low Threshold Spiking, LTS）
        - 内在爆炸（Intrinsically Bursting, IB）
    """
    
    def __init__(
        self,
        input_dim: int,
        neuron_type: str = 'RS',  # Regular Spiking
        a: float = 0.02,
        b: float = 0.2,
        c: float = -65.0,
        d: float = 8.0,
        v_rest: float = -70.0,
        v_threshold: float = 30.0,
        **kwargs
    ):
        """
        初始化Izhikevich神经元
        
        Args:
            input_dim: 输入维度
            neuron_type: 神经元类型 ('RS', 'FS', 'LTS', 'IB')
            a, b, c, d: Izhikevich模型参数
            v_rest: 静息电位
            v_threshold: 发火阈值
            **kwargs: 其他参数
            
        支持的神经元类型参数：
            RS (Regular Spiking): a=0.02, b=0.2, c=-65, d=8
            FS (Fast Spiking): a=0.1, b=0.2, c=-65, d=2
            LTS (Low Threshold Spiking): a=0.02, b=0.25, c=-65, d=2
            IB (Intrinsically Bursting): a=0.02, b=0.2, c=-55, d=4
        """
        nn.Module.__init__(self)
        
        self.input_dim = input_dim
        self.neuron_type = neuron_type
        self.v_threshold = v_threshold
        self.v_rest = v_rest
        
        # 设置参数（由neuron_type确定或用户自定义）
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
        # 突触权重
        self.weight = nn.Parameter(torch.randn(input_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))
        
        # 状态变量
        self.register_buffer('v_mem', torch.full((1,), v_rest))
        self.register_buffer('u_recovery', torch.zeros(1))
        
        # 参数验证和打印
        self._validate_parameters()
    
    def _validate_parameters(self):
        """验证并打印Izhikevich参数"""
        param_dict = {
            'a': self.a, 'b': self.b, 'c': self.c, 'd': self.d
        }
        print(f"Izhikevich {self.neuron_type} 神经元初始化")
        print(f"  参数: {param_dict}")
    
    def forward(
        self,
        x: torch.Tensor,
        dt: float = 0.1
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Izhikevich神经元前向传播
        
        Args:
            x: (batch_size, input_dim) 输入电流
            dt: 时间步长
            
        Returns:
            spike: (batch_size, 1) 脉冲输出
            state: {
                'V': 膜电位,
                'U': 恢复电流,
                'I_ext': 外部输入电流
            }
        """
        # 输入电流
        I_ext = torch.matmul(x, self.weight) + self.bias
        
        # Izhikevich方程
        # dV/dt = 0.04*V² + 5*V + 140 - U + I
        dv = 0.04 * (self.v_mem ** 2) + 5 * self.v_mem + 140 - self.u_recovery + I_ext
        self.v_mem = self.v_mem + dt * dv
        
        # dU/dt = a*(b*V - U)
        du = self.a * (self.b * self.v_mem - self.u_recovery)
        self.u_recovery = self.u_recovery + dt * du
        
        # 发火判断和重置
        spike = (self.v_mem >= self.v_threshold).float()
        
        # 重置
        self.v_mem = self.v_mem * (1 - spike.detach()) + spike.detach() * self.c
        self.u_recovery = self.u_recovery * (1 - spike.detach()) + spike.detach() * (
            self.u_recovery + self.d
        )
        
        return spike, {
            'V': self.v_mem.detach(),
            'U': self.u_recovery.detach(),
            'I_ext': I_ext.detach()
        }
    
    def reset(self):
        """重置神经元状态"""
        self.v_mem.fill_(self.v_rest)
        self.u_recovery.fill_(0.0)


class MultiCompartmentNeuronInterface(nn.Module, NeuronInterface):
    """
    多室神经元模型接口
    
    支持多个细胞室（Compartments）的分布式建模：
        - 胞体（Soma）：决策和集成
        - 树突（Dendrite）：接收突触输入
        - 轴突（Axon）：产生脉冲和输出
    
    每个室有独立的电压、电流方程
    室之间通过轴向电阻连接
    
    方程框架：
        C_m * dV_i/dt = -g_L*(V_i - E_L) + Σ I_synaptic + Σ I_axial + I_ext
        I_axial = g_coupling * (V_j - V_i)  # 室间耦合
    """
    
    def __init__(
        self,
        input_dim: int,
        num_compartments: int = 3,  # soma, dendrite, axon
        C_m: float = 1.0,  # 膜电容
        g_L: float = 0.1,  # 漏电导
        E_L: float = -70.0,  # 漏电位
        g_coupling: float = 0.5,  # 室间耦合电导
        **kwargs
    ):
        """
        初始化多室神经元模型
        
        Args:
            input_dim: 输入维度
            num_compartments: 室数量（默认3：胞体、树突、轴突）
            C_m: 膜电容
            g_L: 漏电导
            E_L: 漏电位
            g_coupling: 室间耦合电导
            **kwargs: 其他参数
            
        室的标准配置：
            compartment 0: Soma（胞体）- 发火点
            compartment 1: Dendrite（树突）- 输入区域
            compartment 2: Axon（轴突）- 输出区域
        """
        nn.Module.__init__(self)
        
        self.input_dim = input_dim
        self.num_compartments = num_compartments
        self.C_m = C_m
        self.g_L = g_L
        self.E_L = E_L
        self.g_coupling = g_coupling
        
        # 每个室的突触权重
        self.compartment_weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim) * 0.01)
            for _ in range(num_compartments)
        ])
        
        self.compartment_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(1))
            for _ in range(num_compartments)
        ])
        
        # 室间耦合系数
        self.coupling_matrix = nn.Parameter(
            self._create_coupling_matrix(num_compartments)
        )
        
        # 状态变量（各室的膜电位）
        self.register_buffer('v_mem', torch.full((num_compartments, 1), E_L))
    
    def _create_coupling_matrix(self, num_compartments: int) -> torch.Tensor:
        """
        创建室间耦合矩阵
        树突 ← → 胞体 ← → 轴突
        """
        matrix = torch.zeros(num_compartments, num_compartments)
        # 相邻室之间的耦合
        for i in range(num_compartments - 1):
            matrix[i, i + 1] = self.g_coupling
            matrix[i + 1, i] = self.g_coupling
        return matrix
    
    def forward(
        self,
        x: torch.Tensor,
        dt: float = 0.1
    ) -> Tuple[torch.Tensor, Dict]:
        """
        多室神经元前向传播
        
        Args:
            x: (batch_size, input_dim) 输入电流
            dt: 时间步长
            
        Returns:
            spike: (batch_size, 1) 胞体发火信号
            state: {
                'V': 各室膜电位,
                'I_ext': 外部电流,
                'I_coupling': 室间耦合电流
            }
        """
        batch_size = x.size(0)
        v_new = self.v_mem.clone()
        
        # 计算各室的膜电位
        I_coupling_all = []
        
        for i in range(self.num_compartments):
            # 外部输入电流（仅胞体接收树突输入）
            if i == 0:  # Soma接收直接输入和树突输入
                I_ext = torch.matmul(x, self.compartment_weights[i]) + self.compartment_bias[i]
            else:
                I_ext = torch.matmul(x, self.compartment_weights[i]) + self.compartment_bias[i]
            
            # 漏电流
            I_leak = self.g_L * (self.v_mem[i] - self.E_L)
            
            # 室间耦合电流
            I_coupling = 0
            for j in range(self.num_compartments):
                if self.coupling_matrix[i, j] > 0:
                    I_coupling += self.coupling_matrix[i, j] * (self.v_mem[j] - self.v_mem[i])
            
            I_coupling_all.append(I_coupling)
            
            # 膜电位更新：C_m * dV/dt = -g_L*(V-E_L) + I_coupling + I_ext
            dv = (I_ext + I_coupling - I_leak) / self.C_m
            v_new[i] = self.v_mem[i] + dt * dv
        
        self.v_mem = v_new
        
        # Soma发火判断
        soma_voltage = self.v_mem[0]  # (1, 1)
        spike = (soma_voltage >= 30).float()  # (1, 1)
        
        # Soma发火时重置（简化的重置）
        self.v_mem[0] = self.v_mem[0] * (1 - spike) + spike * self.E_L
        
        return spike, {
            'V': self.v_mem.detach().clone(),
            'I_coupling': torch.stack(I_coupling_all).detach()
        }
    
    def reset(self):
        """重置所有室的膜电位"""
        self.v_mem.fill_(self.E_L)


class HybridNeuronNetwork(nn.Module):
    """
    混合神经元网络 - 支持多种神经元模型的组合
    
    允许在同一网络中混用不同类型的神经元：
        - LIF（轻量级，高效）
        - Izhikevich（中等复杂度，丰富行为）
        - MultiCompartment（高精度，生物学还原）
        
    提供统一的训练和推断接口
    """
    
    def __init__(
        self,
        layer_configs: list,  # [{type: 'LIF'/'IZH'/'MULTI', ...}, ...]
        input_dim: int
    ):
        """
        Args:
            layer_configs: 各层配置列表
                [{
                    'type': 'LIF' | 'IZH' | 'MULTI',
                    'input_dim': int,
                    'output_dim': int,
                    ...其他参数
                }, ...]
            input_dim: 网络输入维度
        """
        super().__init__()
        self.layer_configs = layer_configs
        self.layers = nn.ModuleList()
        
        # 根据配置构建各层
        current_dim = input_dim
        for config in layer_configs:
            layer_type = config.get('type', 'LIF')
            
            if layer_type == 'LIF':
                from .neuron_models import LeakyIntegrateFire
                layer = LeakyIntegrateFire(
                    input_dim=current_dim,
                    tau=config.get('tau', 2.0),
                    v_threshold=config.get('v_threshold', 1.0)
                )
            elif layer_type == 'IZH':
                layer = IzhikevichNeuronInterface(
                    input_dim=current_dim,
                    neuron_type=config.get('neuron_type', 'RS')
                )
            elif layer_type == 'MULTI':
                layer = MultiCompartmentNeuronInterface(
                    input_dim=current_dim,
                    num_compartments=config.get('num_compartments', 3)
                )
            else:
                raise ValueError(f"Unknown neuron type: {layer_type}")
            
            self.layers.append(layer)
            current_dim = config.get('output_dim', current_dim)
    
    def forward(self, x: torch.Tensor) -> Dict:
        """
        混合网络前向传播
        
        Returns:
            包含各层输出和状态的字典
        """
        states = []
        for layer in self.layers:
            output, state = layer(x)
            states.append(state)
            x = output
        
        return {
            'output': x,
            'states': states
        }


# ==================== 工厂函数 ====================

def create_neuron_layer(
    neuron_type: str,
    input_dim: int,
    **kwargs
) -> nn.Module:
    """
    根据类型创建神经元层的便利函数
    
    Args:
        neuron_type: 'LIF', 'IF', 'HH', 'IZH', 'MULTI'
        input_dim: 输入维度
        **kwargs: 模型特定参数
        
    Returns:
        相应的神经元层
    """
    if neuron_type == 'LIF':
        from .neuron_models import LeakyIntegrateFire
        return LeakyIntegrateFire(input_dim, **kwargs)
    elif neuron_type == 'IF':
        from .neuron_models import IntegrateFire
        return IntegrateFire(input_dim, **kwargs)
    elif neuron_type == 'HH':
        from .neuron_models import HodgkinHuxley
        return HodgkinHuxley(input_dim, **kwargs)
    elif neuron_type == 'IZH':
        return IzhikevichNeuronInterface(input_dim, **kwargs)
    elif neuron_type == 'MULTI':
        return MultiCompartmentNeuronInterface(input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown neuron type: {neuron_type}")
