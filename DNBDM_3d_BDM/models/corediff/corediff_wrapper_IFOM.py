import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from einops import rearrange


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),  # 修改为3D卷积
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        # 使用 3D 卷积转置层（ConvTranspose3d）
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CDHW (C: channels, D: depth, H: height, W: width)
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]

        # 使用 3D padding 来对齐输入
        x1 = F.pad(x1, (diffW // 2, diffW - diffW // 2,
                        diffH // 2, diffH - diffH // 2,
                        diffD // 2, diffD - diffD // 2))
        x = x2 + x1
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        # 使用 3D 卷积层（Conv3d）
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class adjust_net(nn.Module):
    def __init__(self, out_channels=64, middle_channels=32):
        super(adjust_net, self).__init__()

        # 修改卷积层为3D卷积
        self.model = nn.Sequential(
            # 输入通道2变为处理3D数据，depth, height, width
            nn.Conv3d(2, middle_channels, 3, padding=1),  # Conv3d
            nn.ReLU(inplace=True),
            nn.AvgPool3d(2),  # 使用3D池化层

            nn.Conv3d(middle_channels, middle_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(2),

            nn.Conv3d(middle_channels * 2, middle_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(2),

            nn.Conv3d(middle_channels * 4, out_channels * 2, 1, padding=0)
        )

    def forward(self, x):
        # 输入x是一个3D张量，形状为(batch_size, channels, depth, height, width)
        out = self.model(x)
        # 使用自适应池化来减少空间尺寸
        out = F.adaptive_avg_pool3d(out, (1, 1, 1))  # 使用3D自适应池化
        # 分割通道为两个部分
        out1 = out[:, :out.shape[1] // 2]
        out2 = out[:, out.shape[1] // 2:]
        return out1, out2



class EAM_3D(nn.Module):
    """
    3D Edge-Aware Mechanism (EAM)
    """
    def __init__(self, in_channels, rate=4, fft_norm: str = "ortho"):
        super(EAM_3D, self).__init__()
        self.fft_norm = fft_norm


        mid_channels = max(in_channels // rate, 1)
        self.freq_att = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, in_channels, kernel_size=3, padding=1)
        )

        self.alpha = nn.Parameter(torch.ones(1))   # 频域增强系数
        self.gamma = nn.Parameter(torch.ones(1))   # 边缘残差系数

    def forward(self, x):
        """
        x: (B, C, D, H, W)
        """

        X_f = torch.fft.fftn(x, dim=(-3, -2, -1), norm=self.fft_norm)   # complex tensor

        # 2) 频谱幅度 -> 频域注意力
        mag = X_f.abs()                         # (B, C, D, H, W), 实数
        freq_weight = self.freq_att(mag)        # (B, C, D, H, W)
        freq_weight = torch.sigmoid(freq_weight)

        # 3) 在频域中自适应增强高频分量

        Y_f = X_f * (1.0 + self.alpha * freq_weight)

        # 4) IFFT 回到空间域
        y = torch.fft.ifftn(Y_f, dim=(-3, -2, -1), norm=self.fft_norm).real  # (B, C, D, H, W)

        # 5) 显式提取边缘：y 减去局部平滑版本（近似低频），得到高频/边缘
        smooth = F.avg_pool3d(y, kernel_size=3, stride=1, padding=1)   # 近似低频
        edges = y - smooth                                             # 高通成分 ≈ 边缘


        out = x + self.gamma * edges
        return out

class IFOM_Attention_3D(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(IFOM_Attention_3D, self).__init__()

        # **通道注意力 (Channel Attention)**
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // rate),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // rate, in_channels)
        )

        # **空间注意力 (Spatial Attention)**
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // rate, kernel_size=7, padding=3),
            nn.BatchNorm3d(in_channels // rate),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // rate, in_channels, kernel_size=7, padding=3),
            nn.BatchNorm3d(in_channels)
        )

    def forward(self, x):
        b, c, d, h, w = x.shape  # 3D 输入 (Batch, Channels, Depth, Height, Width)

        # **通道注意力**
        x_permute = x.permute(0, 2, 3, 4, 1).reshape(b, -1, c)  # (B, D*H*W, C)
        x_att_permute = self.channel_attention(x_permute).reshape(b, d, h, w, c)
        x_channel_att = x_att_permute.permute(0, 4, 1, 2, 3).sigmoid()  # (B, C, D, H, W)

        # **应用通道注意力**
        x = x * x_channel_att

        # **空间注意力**
        x_spatial_att = self.spatial_attention(x).sigmoid()

        # **应用空间注意力**
        out = x * x_spatial_att

        return out

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        dim = 32
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.inc = nn.Sequential(
            single_conv(in_channels, 64),
            single_conv(64, 64)
        )

        self.down1 = nn.AvgPool3d(2)
        self.mlp1 = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 64)
        )
        self.adjust1 = adjust_net(64)
        self.conv1 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )
       # self.att1 = IFOM_Attention_3D(128)

        self.down2 = nn.AvgPool3d(2)
        self.mlp2 = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 128)
        )
        self.adjust2 = adjust_net(128)
        self.conv2 = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256)
        )
        self.eam2 = EAM_3D(256)
        self.att2 = IFOM_Attention_3D(256)

        self.up1 = up(256)
        self.mlp3 = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 128)
        )
        self.adjust3 = adjust_net(128)
        self.conv3 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )
       # self.att3 = IFOM_Attention_3D(128)

        self.up2 = up(128)
        self.mlp4 = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 64)
        )
        self.adjust4 = adjust_net(64)
        self.conv4 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64)
        )
        self.eam2 = EAM_3D(64)
        self.att4 = IFOM_Attention_3D(64)

        self.outc = outconv(64, out_channels)

    def forward(self, x, t, x_adjust, adjust):
        inx = self.inc(x)
        time_emb = self.time_mlp(t)

        # Down 1
        down1 = self.down1(inx)
        condition1 = self.mlp1(time_emb)
        condition1 = rearrange(condition1, 'b c -> b c 1 1 1')
        if adjust:
            gamma1, beta1 = self.adjust1(x_adjust)
            down1 = down1 + gamma1 * condition1 + beta1
        else:
            down1 = down1 + condition1
        conv1 = self.conv1(down1)
        conv1 = self.att1(conv1)

        # Down 2
        down2 = self.down2(conv1)
        condition2 = self.mlp2(time_emb)
        condition2 = rearrange(condition2, 'b c -> b c 1 1 1')
        if adjust:
            gamma2, beta2 = self.adjust2(x_adjust)
            down2 = down2 + gamma2 * condition2 + beta2
        else:
            down2 = down2 + condition2
        conv2 = self.conv2(down2)
        conv2 = self.att2(conv2)

        # Up 1
        up1 = self.up1(conv2, conv1)
        condition3 = self.mlp3(time_emb)
        condition3 = rearrange(condition3, 'b c -> b c 1 1 1')
        if adjust:
            gamma3, beta3 = self.adjust3(x_adjust)
            up1 = up1 + gamma3 * condition3 + beta3
        else:
            up1 = up1 + condition3
        conv3 = self.conv3(up1)
        conv3 = self.att3(conv3)

        # Up 2
        up2 = self.up2(conv3, inx)
        condition4 = self.mlp4(time_emb)
        condition4 = rearrange(condition4, 'b c -> b c 1 1 1')
        if adjust:
            gamma4, beta4 = self.adjust4(x_adjust)
            up2 = up2 + gamma4 * condition4 + beta4
        else:
            up2 = up2 + condition4
        conv4 = self.conv4(up2)
        conv4 = self.att4(conv4)

        out = self.outc(conv4)
        return out



class Network(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, context=True):
        super(Network, self).__init__()
        self.unet = UNet(in_channels=in_channels, out_channels=out_channels)
        self.context = context
        self.scale_factor = nn.Parameter(torch.ones(1))  # 学习参数控制残差贡献

    def forward(self, x, t, y, x_end, adjust=True):
        # 确保 x 至少有 2 个通道
        if self.context and x.shape[1] > 1:
            x_middle = x[:, 1].unsqueeze(1)  # 取第2通道作为残差
        else:
            x_middle = x

        # 保护性检查，确保 y 和 x_end 维度匹配
        if y.shape[1] != x_end.shape[1]:
            raise ValueError(f"y.shape[1] ({y.shape[1]}) and x_end.shape[1] ({x_end.shape[1]}) must be equal.")

        x_adjust = torch.cat((y, x_end), dim=1)

        out = self.unet(x, t, x_adjust, adjust=adjust)

        # 用可学习的 scale_factor 调整残差影响
        return out + self.scale_factor * x_middle


# WeightNet for one-shot learning framework
class WeightNet(nn.Module):
    def __init__(self, weight_num=10, tau=1.0):
        super(WeightNet, self).__init__()
        self.tau = tau  # 温度参数，避免梯度消失
        init = torch.ones([1, weight_num, 1, 1, 1]) / weight_num  # 适配 3D 数据
        self.weights = nn.Parameter(init)

    def forward(self, x):
        # 计算带温度的 softmax 权重
        weights = F.softmax(self.weights / self.tau, dim=1)
        out = (weights * x).sum(dim=1, keepdim=True)

        return out, weights
