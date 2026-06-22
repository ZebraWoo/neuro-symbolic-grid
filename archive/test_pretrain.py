#!/usr/bin/env python3
"""
预训练系统测试脚本
验证所有模块的正确性
"""

import sys
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """测试所有导入"""
    logger.info("=" * 50)
    logger.info("测试 1: 模块导入")
    logger.info("=" * 50)
    
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        logger.error(f"✗ PyTorch 导入失败: {e}")
        return False
    
    try:
        import pandas as pd
        logger.info(f"✓ Pandas {pd.__version__}")
    except ImportError as e:
        logger.error(f"✗ Pandas 导入失败: {e}")
        return False
    
    try:
        import numpy as np
        logger.info(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        logger.error(f"✗ NumPy 导入失败: {e}")
        return False
    
    try:
        from src.data.load_renewable_dataset import LoadRenewableDataLoader, TimeSeriesDataset
        logger.info("✓ 数据加载模块")
    except ImportError as e:
        logger.error(f"✗ 数据加载模块导入失败: {e}")
        return False
    
    try:
        from src.models.spikformer_pretrain import SpikformerPretrainModel
        logger.info("✓ Spikformer预训练模型")
    except ImportError as e:
        logger.error(f"✗ 预训练模型导入失败: {e}")
        return False
    
    try:
        from src.training.pretrain_training import PretrainingTrainer, RepresentationLearningLoss
        logger.info("✓ 训练模块")
    except ImportError as e:
        logger.error(f"✗ 训练模块导入失败: {e}")
        return False
    
    return True


def test_data_loading():
    """测试数据加载"""
    logger.info("\n" + "=" * 50)
    logger.info("测试 2: 数据加载")
    logger.info("=" * 50)
    
    from src.data.load_renewable_dataset import LoadRenewableDataLoader
    
    data_root = "/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable"
    
    try:
        loader = LoadRenewableDataLoader(data_root)
        logger.info(f"✓ 数据加载器初始化")
        logger.info(f"  发现 {len(loader.zones)} 个区域: {list(loader.zones.keys())}")
        
        # 加载一个区域
        df = loader.load_zone('CAISO_zone_1_')
        logger.info(f"✓ 数据加载成功")
        logger.info(f"  形状: {df.shape}")
        logger.info(f"  特征: {list(df.columns)[:3]}...")
        
        return True
    except Exception as e:
        logger.error(f"✗ 数据加载失败: {e}")
        return False


def test_model():
    """测试模型"""
    logger.info("\n" + "=" * 50)
    logger.info("测试 3: 模型创建和前向传播")
    logger.info("=" * 50)
    
    import torch
    from src.models.spikformer_pretrain import SpikformerPretrainModel
    
    try:
        # 创建模型
        model = SpikformerPretrainModel(
            input_dim=10,
            hidden_dim=128,
            embedding_dim=64,
            num_encoder_layers=2,
        )
        logger.info("✓ 模型创建成功")
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  参数数量: {total_params:,}")
        
        # 测试前向传播
        x = torch.randn(4, 144, 10)  # batch=4, seq_len=144, features=10
        output = model(x)
        
        logger.info("✓ 前向传播成功")
        logger.info(f"  输入: {x.shape}")
        logger.info(f"  嵌入: {output['embedding'].shape}")
        logger.info(f"  编码: {output['encoded'].shape}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_function():
    """测试损失函数"""
    logger.info("\n" + "=" * 50)
    logger.info("测试 4: 损失函数")
    logger.info("=" * 50)
    
    import torch
    from src.training.pretrain_training import RepresentationLearningLoss
    
    try:
        loss_fn = RepresentationLearningLoss(temperature=0.07)
        logger.info("✓ 损失函数创建成功")
        
        # 测试损失计算
        embeddings = torch.randn(8, 64)
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        proximities = torch.rand(8)
        cluster_centers = torch.randn(10, 64)
        
        loss_dict = loss_fn(embeddings, proximities, cluster_centers)
        
        logger.info("✓ 损失计算成功")
        logger.info(f"  总损失: {loss_dict['total'].item():.4f}")
        logger.info(f"  对比损失: {loss_dict['contrastive'].item():.4f}")
        logger.info(f"  分离损失: {loss_dict['separation'].item():.4f}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 损失函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    """测试数据集"""
    logger.info("\n" + "=" * 50)
    logger.info("测试 5: 时间序列数据集")
    logger.info("=" * 50)
    
    from src.data.load_renewable_dataset import LoadRenewableDataLoader, TimeSeriesDataset
    
    try:
        data_root = "/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable"
        loader = LoadRenewableDataLoader(data_root)
        df = loader.load_zone('CAISO_zone_1_')
        
        # 创建数据集
        dataset = TimeSeriesDataset(df, seq_len=144, stride=36, normalize='zscore')
        logger.info("✓ 数据集创建成功")
        logger.info(f"  样本数: {len(dataset)}")
        
        # 获取样本
        x, info = dataset[0]
        logger.info("✓ 样本获取成功")
        logger.info(f"  样本形状: {x.shape}")
        logger.info(f"  时间范围: {info['start_time']} 到 {info['end_time']}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    logger.info("\n" + "🧪" * 25)
    logger.info("Spikformer预训练系统测试")
    logger.info("🧪" * 25 + "\n")
    
    tests = [
        ("模块导入", test_imports),
        ("数据加载", test_data_loading),
        ("模型", test_model),
        ("损失函数", test_loss_function),
        ("数据集", test_dataset),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            logger.error(f"✗ {name} 测试异常: {e}")
            results.append((name, False))
    
    # 总结
    logger.info("\n" + "=" * 50)
    logger.info("测试总结")
    logger.info("=" * 50)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        logger.info("\n🎉 所有测试通过！系统已就绪。")
        logger.info("\n下一步：运行以下命令启动预训练")
        logger.info("  python train_pretrain.py --zones CAISO_zone_1_ CAISO_zone_2_ --num-epochs 5")
    else:
        logger.error(f"\n❌ {total - passed} 个测试失败。请检查环境配置。")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
