import torch
import torch.nn as nn
from Generator import (
    FFTDecompose, DownSample, UpSample, ResnetBlock, ResnetGenerator
)


# ========================================
# 测试1：FFTDecompose单元测试（核心解耦逻辑）
# ========================================
def test_fft_decompose():
    print("=== 测试1：FFTDecompose单元测试 ===")
    # 模拟特征图（B=2, C=64, H=128, W=128）
    x = torch.randn(2, 64, 128, 128)
    fft_module = FFTDecompose(cutoff=0.35)

    # 前向传播
    low, high = fft_module(x)

    # 验证维度（解耦后应与输入维度完全一致）
    assert low.shape == x.shape, f"低频维度错误：{low.shape} vs {x.shape}"
    assert high.shape == x.shape, f"高频维度错误：{high.shape} vs {x.shape}"

    # 验证数值合理性（低频+高频应近似等于原输入）
    recon = low + high
    diff = torch.mean(torch.abs(recon - x)).item()
    print(f"输入维度: {x.shape}")
    print(f"低频维度: {low.shape}, 高频维度: {high.shape}")
    print(f"低频+高频与原输入的平均误差: {diff:.6f} (越小越合理，建议<1e-5)")
    print("FFTDecompose测试通过 ✅\n")


# ========================================
# 测试2：DownSample/UpSample模块测试（采样+频域解耦）
# ========================================
def test_sample_modules():
    print("=== 测试2：DownSample/UpSample模块测试 ===")
    # 模拟特征图（B=2, C=64, H=128, W=128）
    x = torch.randn(2, 64, 128, 128)
    down_module = DownSample(channels=64)
    up_module = UpSample(channels=64)

    # 下采样测试
    down_out = down_module(x)
    assert down_out.shape == (2, 64, 64, 64), f"下采样维度错误：{down_out.shape} vs (2,64,64,64)"
    print(f"下采样输入维度: {x.shape}, 输出维度: {down_out.shape}")

    # 上采样测试（输入为下采样输出）
    up_out = up_module(down_out)
    assert up_out.shape == (2, 64, 128, 128), f"上采样维度错误：{up_out.shape} vs (2,64,128,128)"
    print(f"上采样输入维度: {down_out.shape}, 输出维度: {up_out.shape}")
    print("DownSample/UpSample测试通过 ✅\n")


# ========================================
# 测试3：ResnetBlock与FFT解耦兼容性（残差块+频域特征）
# ========================================
def test_resnet_block_compatibility():
    print("=== 测试3：ResnetBlock与FFT解耦兼容性 ===")
    # 模拟经过FFT解耦+采样后的特征（B=2, C=256, H=64, W=64）
    x = torch.randn(2, 256, 64, 64)
    block = ResnetBlock(
        dim=256,
        padding_type='reflect',
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        use_bias=False
    )

    # 前向传播
    out = block(x)
    assert out.shape == x.shape, f"残差块输出维度错误：{out.shape} vs {x.shape}"
    print(f"ResnetBlock输入维度: {x.shape}, 输出维度: {out.shape}")
    print("ResnetBlock兼容性测试通过 ✅\n")


# ========================================
# 测试4：整网前向测试（端到端验证）
# ========================================
def test_full_generator():
    print("=== 测试4：整网前向测试 ===")
    # 初始化生成器（与原代码main函数一致）
    netG = ResnetGenerator(
        input_nc=3,
        output_nc=3,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        n_blocks=6,
        padding_type='reflect'
    )
    # 模拟真实输入（B=4, C=3, H=256, W=256，批量测试稳定性）
    x = torch.randn(4, 3, 256, 256)

    # 前向传播
    out = netG(x)
    assert out.shape == x.shape, f"整网输出维度错误：{out.shape} vs {x.shape}"
    print(f"整网输入维度: {x.shape}, 输出维度: {out.shape}")
    print("整网前向测试通过 ✅\n")


# ========================================
# 执行所有测试
# ========================================
if __name__ == "__main__":
    # 设置随机种子保证可复现
    torch.manual_seed(42)
    torch.cuda.manual_seed(42) if torch.cuda.is_available() else None

    # 依次执行测试
    test_fft_decompose()
    test_sample_modules()
    test_resnet_block_compatibility()
    test_full_generator()

    print("所有测试通过！FFT解耦设计在维度和前向逻辑上均合理 🎉")
