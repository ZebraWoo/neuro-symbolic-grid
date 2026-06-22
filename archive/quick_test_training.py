#!/usr/bin/env python
"""
超轻量级训练测试 - 不加载实际数据，直接测试模型训练流程
用于快速验证所有修复是否有效
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.insert(0, '/home/wuzuoxu/electric-power-regulation')

print("=" * 70)
print("🔬 超轻量级训练测试 (合成数据)")
print("=" * 70)

# 参数
batch_size = 2
seq_len = 72
input_dim = 11
hidden_dim = 32
num_epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\n📋 配置:")
print(f"   设备: {device}")
print(f"   Batch: {batch_size}, Seq Len: {seq_len}, Input Dim: {input_dim}")
print(f"   Hidden Dim: {hidden_dim}, Epochs: {num_epochs}")

# 创建合成数据（不需要加载真实数据）
print(f"\n1️⃣  创建合成数据...")
X = torch.randn(100, seq_len, input_dim)
dataset = TensorDataset(X)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"   ✅ 数据加载器: {len(dataloader)} batches")

# 导入模型
print(f"\n2️⃣  导入模型...")
try:
    from src.models.spikformer_pretrain import SpikformerPretrainModel
    
    model = SpikformerPretrainModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embedding_dim=32,
        num_encoder_layers=1,
        num_heads=2
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ 模型加载成功")
    print(f"   ✅ 参数数量: {num_params:,}")
except Exception as e:
    print(f"   ❌ 模型加载失败: {e}")
    sys.exit(1)

# 设置优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# 训练循环
print(f"\n3️⃣  开始训练...")
model.train()

for epoch in range(num_epochs):
    total_loss = 0.0
    
    for batch_idx, (x,) in enumerate(dataloader):
        x = x.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(x)
        
        # 损失计算（修复后的版本）
        encoded = output['encoded']  # (batch, seq_len, hidden_dim)
        encoded_mean = encoded.mean(dim=1, keepdim=True)
        target = encoded_mean.expand_as(encoded)
        loss = loss_fn(encoded, target)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"   Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    print(f"\n✅ Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

print("\n" + "=" * 70)
print("✨ 训练测试成功!")
print("=" * 70)
print("\n📝 验证通过项:")
print("   ✅ 模型前向传播")
print("   ✅ 损失函数计算（修复后）")
print("   ✅ 反向传播")
print("   ✅ 梯度裁剪")
print("   ✅ 参数更新")

print("\n🎉 可以开始实际训练:")
print("   bash test_ultra_lightweight.sh")
print("=" * 70)
