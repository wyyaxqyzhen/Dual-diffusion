import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from math import exp
from torch.autograd import Variable


def compute_measure(y, pred, data_range):
    psnr = compute_PSNR(pred, y, data_range)
    ssim = compute_SSIM(pred, y, data_range)
    rmse = compute_RMSE(pred, y)
    return psnr, ssim, rmse


def compute_MSE(img1, img2):
    return ((img1/1.0 - img2/1.0) ** 2).mean()


def compute_RMSE(img1, img2):
    # img1 = img1 * 2000 / 255 - 1000
    # img2 = img2 * 2000 / 255 - 1000
    if type(img1) == torch.Tensor:
        return torch.sqrt(compute_MSE(img1, img2)).item()
    else:
        return np.sqrt(compute_MSE(img1, img2))


def compute_PSNR(img1, img2, data_range):
    if type(img1) == torch.Tensor:
        mse_ = compute_MSE(img1, img2)
        return 10 * torch.log10((data_range ** 2) / mse_).item()
    else:
        mse_ = compute_MSE(img1, img2)
        return 10 * np.log10((data_range ** 2) / mse_)


# def compute_SSIM(img1, img2, data_range, window_size=11, channel=1, size_average=True):
#     # referred from https://github.com/Po-Hsun-Su/pytorch-ssim
#     if len(img1.shape) == 2:
#         h, w = img1.shape
#         if type(img1) == torch.Tensor:
#             img1 = img1.view(1, 1, h, w)
#             img2 = img2.view(1, 1, h, w)
#         else:
#             img1 = torch.from_numpy(img1[np.newaxis, np.newaxis, :, :])
#             img2 = torch.from_numpy(img2[np.newaxis, np.newaxis, :, :])
#     window = create_window(window_size, channel)
#     window = window.type_as(img1)
#
#     mu1 = F.conv2d(img1, window, padding=window_size//2)
#     mu2 = F.conv2d(img2, window, padding=window_size//2)
#     mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
#     mu1_mu2 = mu1*mu2
#
#     sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2) - mu1_sq
#     sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2) - mu2_sq
#     sigma12 = F.conv2d(img1*img2, window, padding=window_size//2) - mu1_mu2
#
#     C1, C2 = (0.01*data_range)**2, (0.03*data_range)**2
#     #C1, C2 = 0.01**2, 0.03**2
#
#     ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
#     # if size_average:
#     #     return ssim_map.mean().item()
#     # else:
#     #     return ssim_map.mean(1).mean(1).mean(1).item()
#     if size_average:
#         return ssim_map.mean().item()
#     else:
#         return ssim_map.mean(1).mean(1).mean(1).item()
#
#
# def gaussian(window_size, sigma):
#     gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
#     return gauss / gauss.sum()
#
#
# def create_window(window_size, channel):
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
#     return window





# def compute_SSIM(img1, img2, data_range, window_size=11, channel=1, size_average=True):
#     """
#     计算 3D 图像的 SSIM 指标。
#     参数:
#         img1, img2: 输入的 3D 图像，形状为 (N, C, D, H, W)。
#         data_range: 图像的动态范围（例如，16 位图像为 65535）。
#         window_size: 高斯窗口的大小，建议为 11。
#         channel: 图像的通道数，默认为 1。
#         size_average: 如果为 True，返回所有 SSIM 值的均值；否则返回局部 SSIM。
#     返回:
#         SSIM 值。
#     """
#     if len(img1.shape) == 3:  # 处理单个 3D 图像 (D, H, W)
#         d, h, w = img1.shape
#         if isinstance(img1, torch.Tensor):
#             img1 = img1.view(1, 1, d, h, w)
#             img2 = img2.view(1, 1, d, h, w)
#         else:
#             img1 = torch.from_numpy(img1[np.newaxis, np.newaxis, :, :, :])
#             img2 = torch.from_numpy(img2[np.newaxis, np.newaxis, :, :, :])

#     window = create_3d_window(window_size, channel)
#     window = window.type_as(img1)

#     # 均值计算
#     mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
#     mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

#     # 方差与协方差计算
#     mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
#     sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
#     sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

#     # SSIM 公式
#     C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

#     if size_average:
#         return ssim_map.mean().item()
#     else:
#         return ssim_map  # 返回局部 SSIM map


# def gaussian(window_size, sigma):
#     """
#     创建 1D 高斯核。
#     """
#     gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
#     return gauss / gauss.sum()


# def create_3d_window(window_size, channel):
#     """
#     创建 3D 高斯窗口。
#     """
#     # 创建 1D 高斯核
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # (window_size, 1)
#     # 创建 2D 高斯核
#     _2D_window = _1D_window.mm(_1D_window.t()).float()  # (window_size, window_size)
#     # 创建 3D 高斯核
#     _3D_window = torch.stack([_2D_window] * window_size, dim=0)  # (window_size, window_size, window_size)
#     _3D_window = _3D_window / _3D_window.sum()  # 归一化
#     # 扩展维度并匹配目标形状
#     window = _3D_window.unsqueeze(0).unsqueeze(0)  # (1, 1, window_size, window_size, window_size)
#     window = window.expand(channel, 1, window_size, window_size, window_size).contiguous()  # (channel, 1, window_size, window_size, window_size)
#     return Variable(window)

import torch
import torch.nn.functional as F
from math import exp
import numpy as np


def compute_SSIM(img1, img2, data_range, window_size=11, channel=1, size_average=True):
    # Convert from numpy to torch if needed
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1)
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2)

    # Ensure float32 for computation and move to same device
    img1 = img1.float()
    img2 = img2.float()

    # Auto reshape if input is (D, H, W)
    if img1.ndim == 3:
        img1 = img1.view(1, 1, *img1.shape)
        img2 = img2.view(1, 1, *img2.shape)

    # Dynamically adjust window size if input is smaller
    _, _, D, H, W = img1.shape
    min_size = min(D, H, W)
    if window_size > min_size:
        window_size = min_size if min_size % 2 == 1 else min_size - 1  # Keep window size odd

    # Move window to same device
    device = img1.device
    window = create_3d_window(window_size, channel).to(dtype=img1.dtype, device=device)

    padding = window_size // 2
    mu1 = F.conv3d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv3d(img2, window, padding=padding, groups=channel)

    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv3d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item() if size_average else ssim_map


def gaussian(window_size, sigma):
    center = window_size // 2
    gauss = torch.tensor([exp(-(x - center) ** 2 / (2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_3d_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window @ _1D_window.t()
    _3D_window = torch.stack([_2D_window] * window_size, dim=0)
    _3D_window = _3D_window / _3D_window.sum()
    window = _3D_window.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, D, H, W)
    window = window.expand(channel, 1, window_size, window_size, window_size).contiguous()
    return window
