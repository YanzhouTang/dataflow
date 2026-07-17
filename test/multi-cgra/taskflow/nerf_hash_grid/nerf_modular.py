#!/cluster/home/tangyz/.conda/envs/torch-mlir-env/bin/python
"""
模块化 NeRF 实现 - 为 Taskflow 分区优化

设计思路：
- 每个子模块作为独立的 nn.Module
- 在 forward 中显式调用子模块
- 转换后每个子模块调用会成为一个 Taskflow task
- 便于硬件分区 (partition by target)
"""

import torch
import torch.nn as nn

# 使用 TOSA 兼容版本（避免 linspace 等问题）
try:
    from nerf_hash_grid_tosa_compatible import RaySampler, HashGridEncoder, NeRFMLP
    print("✓ 使用 TOSA 兼容版本")
except ImportError:
    from nerf_hash_grid import RaySampler, HashGridEncoder, NeRFMLP
    print("⚠ 使用原始版本（可能有问题）")


class ModularNeRF(nn.Module):
    """
    模块化 NeRF - 便于 Taskflow 任务分区
    
    三个独立的子模块：
    1. ray_sampler_module  → CPU/DOE (轻量级)
    2. hash_encoder_module → DOE (内存密集)
    3. mlp_module          → CGRA (计算密集)
    """
    
    def __init__(self, num_samples=16, num_levels=2, features_per_level=2, hidden_dim=32):
        super().__init__()
        
        # 子模块 1: Ray Sampling
        self.ray_sampler_module = RaySampler(num_samples=num_samples)
        
        # 子模块 2: Hash Encoding
        self.hash_encoder_module = HashGridEncoder(
            num_levels=num_levels,
            features_per_level=features_per_level,
            log2_hashmap_size=8
        )
        
        # 子模块 3: MLP
        self.mlp_module = NeRFMLP(
            input_dim=num_levels * features_per_level,
            hidden_dim=hidden_dim,
            num_layers=2
        )
    
    def forward(self, rays_o, rays_d):
        """
        显式的模块调用 - 每个调用会成为一个 Taskflow task
        
        期望的 Taskflow 结构:
          Task_0 (ray_sampler) → channel → Task_1 (hash_encoder) 
                                           → channel → Task_2 (mlp)
        """
        # Task 1: Sample positions along rays
        # 目标硬件: CPU (轻量级计算)
        positions = self.ray_sampler_module(rays_o, rays_d)
        
        # Task 2: Hash grid encoding  
        # 目标硬件: DOE (内存密集型，哈希表查找)
        encoded_features = self.hash_encoder_module(positions)
        
        # Task 3: MLP inference
        # 目标硬件: CGRA (计算密集型，矩阵乘法)
        density, rgb = self.mlp_module(encoded_features, rays_d)
        
        return density, rgb


def test_modular_nerf():
    """测试模块化 NeRF"""
    print("=" * 70)
    print("测试模块化 NeRF")
    print("=" * 70)
    
    device = torch.device('cpu')
    
    # 创建模型
    model = ModularNeRF(
        num_samples=16,
        num_levels=2,
        features_per_level=2,
        hidden_dim=32
    )
    model.eval()
    
    # 测试输入
    rays_o = torch.randn(2, 3, device=device)
    rays_d = torch.randn(2, 3, device=device)
    
    print(f"\n输入:")
    print(f"  rays_o: {rays_o.shape}")
    print(f"  rays_d: {rays_d.shape}")
    
    # 前向传播
    with torch.no_grad():
        density, rgb = model(rays_o, rays_d)
    
    print(f"\n输出:")
    print(f"  density: {density.shape}")
    print(f"  rgb: {rgb.shape}")
    
    print(f"\n✓ 模块化 NeRF 运行成功!")
    
    return model, rays_o, rays_d


def convert_to_linalg():
    """转换模块化 NeRF 到 Linalg"""
    print("\n" + "=" * 70)
    print("转换模块化 NeRF → Linalg")
    print("=" * 70)
    
    import torch_mlir
    
    # 测试模型
    model, rays_o, rays_d = test_modular_nerf()
    
    try:
        print("\n正在转换为 Linalg MLIR...")
        print("  保留模块结构: 3 个子模块调用")
        
        mlir_module = torch_mlir.compile(
            model,
            (rays_o, rays_d),
            output_type=torch_mlir.OutputType.LINALG_ON_TENSORS,
            use_tracing=True
        )
        
        output_file = "nerf_modular_linalg.mlir"
        with open(output_file, 'w') as f:
            f.write(str(mlir_module))
        
        print(f"\n✓ 成功! Linalg MLIR 已保存到: {output_file}")
        print(f"  文件大小: {len(str(mlir_module)):,} 字符")
        
        # 检查是否保留了模块结构
        mlir_str = str(mlir_module)
        num_funcs = mlir_str.count('func.func')
        num_calls = mlir_str.count('func.call') + mlir_str.count('call @')
        
        print(f"\n结构分析:")
        print(f"  函数定义数: {num_funcs}")
        print(f"  函数调用数: {num_calls}")
        
        if num_calls >= 3:
            print(f"  ✓ 保留了模块调用结构!")
            print(f"  转换到 Taskflow 后会生成 {num_calls} 个 task")
        else:
            print(f"  ⚠ 可能被内联了（函数调用数少于预期）")
            print(f"  这不影响功能，但失去了模块化优势")
        
        print("\n" + "=" * 70)
        print("下一步: 编译到 Taskflow")
        print("=" * 70)
        
        print(f"\n# 完整编译流程")
        print(f"mlir-neura-opt {output_file} \\")
        print(f"  --one-shot-bufferize{{bufferize-function-boundaries=1}} \\")
        print(f"  --pass-pipeline='func.func(convert-linalg-to-affine-loops)' \\")
        print(f"  --convert-affine-to-taskflow \\")
        print(f"  -o nerf_taskflow_modular.mlir")
        
        print(f"\n# 然后添加硬件标注")
        print(f"手动编辑 nerf_taskflow_modular.mlir，添加 target 属性:")
        print(f'  taskflow.task "ray_sampler" <{{target="CPU"}}> {{ ... }}')
        print(f'  taskflow.task "hash_encoder" <{{target="DOE"}}> {{ ... }}')
        print(f'  taskflow.task "nerf_mlp" <{{target="CGRA"}}> {{ ... }}')
        
        print(f"\n# 运行 partition pass")
        print(f"mlir-neura-opt nerf_taskflow_modular.mlir \\")
        print(f"  --partition-taskflow-by-target \\")
        print(f"  -o nerf_partitioned.mlir")
        
        print(f"\n这会自动标注跨硬件的 channel!")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = convert_to_linalg()
    sys.exit(0 if success else 1)
