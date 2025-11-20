import torch
from torch import nn
import math

# 提取 shape
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# 均值退化算子中的 a 和 1-a
def linear_alpha_schedule(timesteps):
    steps = timesteps
    alphas_cumprod = 1 - torch.linspace(0, steps, steps) / timesteps
    return torch.clip(alphas_cumprod, 0, 0.999)

# 实例化扩散模型
class Diffusion(nn.Module):
    def __init__(self,
                 denoise_fn=None,
                 image_size=(36, 36, 36),  # For 3D images, specify depth, height, width
                 channels=1,
                 timesteps=5,
                 context=False,
                 ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.num_timesteps = int(timesteps)


        alphas_cumprod = linear_alpha_schedule(timesteps)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('one_minus_alphas_cumprod', 1. - alphas_cumprod)

    # 均值退化算子
    def q_sample(self, x_start, x_end, t):
        return (
            extract(self.alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.one_minus_alphas_cumprod, t, x_start.shape) * x_end
        )

    # 用于中间的加噪去噪采样
    def get_x2_bar_from_xt(self, x1_bar, xt, t):
        return (
            (xt - extract(self.alphas_cumprod, t, x1_bar.shape) * x1_bar) /
            extract(self.one_minus_alphas_cumprod, t, x1_bar.shape)
        )

    # 加噪去噪采样
    @torch.no_grad()
    def sample(self, batch_size=4, img=None, t=None, sampling_routine='ddim', n_iter=1, start_adjust_iter=1):

        self.denoise_fn.eval()
        if t is None:
            t = self.num_timesteps

        # Initialize variables
        noise = img
        x1_bar = img
        direct_recons = []
        imstep_imgs = []

        if sampling_routine == 'ddim':
            while t:
                step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)

                # Pass input through denoise function
                x1_bar = self.denoise_fn(img, step, x1_bar, noise,
                                         adjust=(t != self.num_timesteps and n_iter >= start_adjust_iter))

                # Compute x2_bar from x_t
                x2_bar = self.get_x2_bar_from_xt(x1_bar, img, step)

                # Update xt_bar
                xt_bar = x1_bar
                if t != 0:
                    xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

                # Compute xt_sub1_bar
                xt_sub1_bar = x1_bar
                if t - 1 != 0:
                    step2 = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
                    xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

                # Update image
                img = img - xt_bar + xt_sub1_bar

                # Save intermediate results
                direct_recons.append(x1_bar)
                imstep_imgs.append(img)
                t = t - 1

        # Convert results to tensors and return
        return img.clamp(0., 1.), torch.stack(direct_recons), torch.stack(imstep_imgs)

    # 前向传播  去噪分为两个阶段，代表着文中的误差调制模块
    def forward(self, x, y, n_iter, only_adjust_two_step=False, start_adjust_iter=1):
        import numpy as np

        # 加载 .npy 文件
        file_path = "data_preprocess/gen_data/spect_16cg_npy/P001_16cg_tif.npy"  # 替换为你的文件路径
        data = np.load(file_path)

        # 打印文件的形状和维度信息
        print(f"Shape of the file '{file_path}': {data.shape}")
        print(f"Number of dimensions: {data.ndim}")

        # 检查是否为 3D
        if data.ndim == 3:
            print(f"The file '{file_path}' is 3D.")
        else:
            print(f"The file '{file_path}' is not 3D.")

        # 解包形状
        b, c, d, h, w = y.shape
        device = y.device

        # 确保 img_size 是三元组
        img_size = self.image_size if isinstance(self.image_size, tuple) else (self.image_size,) * 3

        print(f"Input tensor shape: {y.shape}")  # 调试打印
        print(f"Device: {device}, Image size: {img_size}")
        print(f"Expected img_size: {img_size}, Actual shape: {(d, h, w)}")
        assert (d, h, w) == img_size, f"Depth, height, and width of image must be {img_size}"

        # 随机生成时间步 t
        t_single = torch.randint(0, self.num_timesteps, (1,), device=device).long()
        t = t_single.repeat((b,))

        # 构造 x_mix
        x_end = x
        x_mix = self.q_sample(x_start=y, x_end=x_end, t=t)

        # stage I
        adjust = not (only_adjust_two_step or n_iter < start_adjust_iter or t[0] == self.num_timesteps - 1)
        x_recon = self.denoise_fn(x_mix, t, y, x_end, adjust=adjust)

        # stage II
        if n_iter >= start_adjust_iter and t_single.item() >= 1:
            t_sub1 = torch.clamp(t - 1, min=0)
            x_mix_sub1 = self.q_sample(x_start=x_recon, x_end=x_end, t=t_sub1)
            x_recon_sub1 = self.denoise_fn(x_mix_sub1, t_sub1, x_recon, x_end, adjust=True)
        else:
            x_recon_sub1, x_mix_sub1 = x_recon, x_mix

        return x_recon, x_mix, x_recon_sub1, x_mix_sub1
