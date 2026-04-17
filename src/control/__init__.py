"""
电网自主调控系统 - 控制模块
"""

from .neuron_models import (
    LeakyIntegrateFire,
    IntegrateFire,
    HodgkinHuxley,
    StaticSynapse,
    DynamicSynapse,
    SynapticPlasticity,
    NEURON_MODELS,
    SYNAPSE_MODELS
)

from .multimodal_control_network import (
    MultimodalEmbedding,
    SpikeFormerControlBlock,
    MultimodalControlNetwork,
    ControlPretrainingLoss
)

from .advanced_neuron_models import (
    NeuronInterface,
    IzhikevichNeuronInterface,
    MultiCompartmentNeuronInterface,
    HybridNeuronNetwork,
    create_neuron_layer
)

__version__ = "1.0.0"
__all__ = [
    # 基础神经元
    "LeakyIntegrateFire",
    "IntegrateFire",
    "HodgkinHuxley",
    # 突触模型
    "StaticSynapse",
    "DynamicSynapse",
    "SynapticPlasticity",
    # 模型库
    "NEURON_MODELS",
    "SYNAPSE_MODELS",
    # 多模态网络
    "MultimodalEmbedding",
    "SpikeFormerControlBlock",
    "MultimodalControlNetwork",
    "ControlPretrainingLoss",
    # 高级模型
    "NeuronInterface",
    "IzhikevichNeuronInterface",
    "MultiCompartmentNeuronInterface",
    "HybridNeuronNetwork",
    "create_neuron_layer",
]
