import torch
import torch.nn as nn


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        x, weight, bias, eps = args
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class FreqSplitMLP(nn.Module):
    """修复版：对每个特征通道独立做FFT高频/低频处理，保留通道数"""
    def __init__(self, nc, expand=2, low_ratio=0.2):
        super(FreqSplitMLP, self).__init__()
        self.nc = nc  # 特征图通道数（如64/128），不再强制单通道
        self.expand = expand
        self.low_ratio = low_ratio  # 低频区域占比

        # 高频处理MLP（输入输出通道数=nc）
        self.high_mlp = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0)
        )
        # 低频处理MLP（输入输出通道数=nc）
        self.low_mlp = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0)
        )

    def split_freq(self, x_freq, H, W):
        """分离频域的高频和低频分量（适配rfft2的输出维度）"""
        B, C, H_fft, W_fft = x_freq.shape  # W_fft = W//2 + 1
        # 计算低频区域的尺寸（中心正方形）
        low_H = int(H * self.low_ratio)
        low_W = int(W * self.low_ratio)
        # 中心坐标
        center_H, center_W = H // 2, W // 2
        # 低频掩码：中心区域为1，其余为0（适配rfft2的W维度）
        low_mask = torch.zeros((B, C, H_fft, W_fft), dtype=torch.float32, device=x_freq.device)
        # 处理H维度的掩码范围
        h_start = max(0, center_H - low_H // 2)
        h_end = min(H_fft, center_H + low_H // 2)
        # 处理W维度的掩码范围（适配rfft2的W_fft）
        w_start = max(0, center_W - low_W // 2)
        w_end = min(W_fft, center_W + low_W // 2)
        low_mask[:, :, h_start:h_end, w_start:w_end] = 1.0
        # 高频掩码
        high_mask = 1.0 - low_mask
        # 分离高频、低频
        low_freq = x_freq * low_mask
        high_freq = x_freq * high_mask
        return low_freq, high_freq

    def forward(self, x):
        B, C, H, W = x.shape
        # 断言通道数匹配：确保输入通道数=MLP的通道数
        assert C == self.nc, f"输入通道数{C}与MLP初始化通道数{self.nc}不匹配"

        # 步骤1：对每个通道做实数FFT（保留通道数）
        x_freq = torch.fft.rfft2(x, dim=(-2, -1), norm='backward')  # B×C×H×(W//2+1)
        x_freq_shift = torch.fft.fftshift(x_freq, dim=(-2, -1))  # 中心化频域

        # 步骤2：分离高频、低频（适配rfft2维度）
        low_freq, high_freq = self.split_freq(x_freq_shift, H, W)

        # 步骤3：分别处理幅值（相位保持不变）
        # 低频处理
        low_mag = torch.abs(low_freq)  # B×C×H×(W//2+1)
        low_mag_processed = self.low_mlp(low_mag)  # 通道数保持C
        low_pha = torch.angle(low_freq)
        low_freq_processed = low_mag_processed * torch.exp(1j * low_pha)

        # 高频处理
        high_mag = torch.abs(high_freq)
        high_mag_processed = self.high_mlp(high_mag)  # 通道数保持C
        high_pha = torch.angle(high_freq)
        high_freq_processed = high_mag_processed * torch.exp(1j * high_pha)

        # 步骤4：融合高频、低频
        x_freq_fused = low_freq_processed + high_freq_processed
        x_freq_fused = torch.fft.ifftshift(x_freq_fused, dim=(-2, -1))  # 逆中心化

        # 步骤5：逆FFT回到空间域（保留通道数）
        x_out = torch.fft.irfft2(x_freq_fused, s=(H, W), dim=(-2, -1), norm='backward')  # B×C×H×W

        return x_out


class Branch(nn.Module):
    """EBlock的分支模块（空洞卷积）"""

    def __init__(self, c, DW_Expand, dilation=1):
        super().__init__()
        self.dw_channel = DW_Expand * c
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3,
                      padding=dilation, stride=1, groups=self.dw_channel, bias=True, dilation=dilation)
        )

    def forward(self, input):
        return self.branch(input)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2



class EBlock(nn.Module):
    """修复后的EBlock：保留特征通道数，独立处理每个通道"""

    def __init__(self, c, DW_Expand=2, dilations=[1, 2], extra_depth_wise=False):
        super().__init__()
        self.dw_channel = DW_Expand * c
        # 可选的深度卷积
        self.extra_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True,
                                    dilation=1) if extra_depth_wise else nn.Identity()
        # 1×1卷积升维
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)
        # 多 dilation 分支
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(c, DW_Expand, dilation=dilation))
        # 通道注意力
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0,
                      stride=1, groups=1, bias=True, dilation=1),
        )
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)
        # 归一化
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # 修复后的频域处理模块（通道数=c）
        self.freq = FreqSplitMLP(nc=c, expand=2, low_ratio=0.2)
        # 可学习的缩放系数
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        # 第一步：空洞卷积分支 + 门控 + 注意力
        x = self.norm1(inp)
        x = self.conv1(self.extra_conv(x))
        z = 0
        for branch in self.branches:
            z += branch(x)
        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x

        # 第二步：频域（高频/低频）处理 + 残差
        x_step2 = self.norm2(y)
        x_freq = self.freq(x_step2)  # 现在通道数匹配，不会报错
        x = y * x_freq
        x = y + x * self.gamma

        return x
