import torch
import os.path as osp
import os
import time
from torchvision.utils import save_image
import torch.distributed as dist
import inspect
from utils.ops import reduce_tensor, load_network
import pandas as pd
import pydicom
import tifffile as tiff
import glob

def get_varname(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


class LoggerX(object):

    def __init__(self, save_root):
        # 定义模型保存和图像保存的目录路径
        self.models_save_dir = osp.join(save_root, 'save_models')
        self.images_save_dir = osp.join(save_root, 'save_images')
        self.tif_save_dir = osp.join(save_root,'save_tif') 
        self.tiff_save_dif = osp.join(save_root,'save_tiff')
        self.IMA_save_dir = osp.join(save_root, 'save_IMA')
        self.metrics_save_dir = osp.join(save_root, 'save_metrics')
        self.metrics_df = pd.DataFrame(columns=['epoch', 'PSNR', 'SSIM', 'RMSE'])

        # 创建保存目录，如果目录已经存在则不创建
        os.makedirs(self.models_save_dir, exist_ok=True)
        os.makedirs(self.images_save_dir, exist_ok=True)
        os.makedirs(self.IMA_save_dir, exist_ok=True)
        os.makedirs(self.metrics_save_dir, exist_ok=True)
        #os.makedirs(self.tif_save_tif,exist_ok=True)
        self._modules = []
        self._module_names = []
        self.world_size = 1
        self.local_rank = 0
        # 存储训练过程的 loss
        self.train_losses = []
        self.val_losses = []
        self.val_best_loss = float('inf')


    @property
    def modules(self):
        return self._modules

    @property
    def module_names(self):
        return self._module_names

    @modules.setter
    def modules(self, modules):
        for i in range(len(modules)):
            self._modules.append(modules[i])
            self._module_names.append(get_varname(modules[i]))

    def checkpoints(self, epoch, max_to_keep=5):
        if self.local_rank != 0:
            return

        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]

            # 保存新的检查点
            save_path = osp.join(self.models_save_dir, f'{module_name}-{epoch}')
            torch.save(module.state_dict(), save_path)
            print(f"Checkpoint saved: {save_path}")

            # 清理旧的检查点
            # 查找该模块的所有检查点文件
            checkpoint_files = sorted(
                glob.glob(osp.join(self.models_save_dir, f'{module_name}-*')),
                key=os.path.getmtime
            )

            # 如果超过 max_to_keep，删除最旧的文件
            if len(checkpoint_files) > max_to_keep:
                for old_file in checkpoint_files[:-max_to_keep]:
                    os.remove(old_file)
                    print(f"Deleted old checkpoint: {old_file}")

    def load_checkpoints(self, epoch):
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            module.load_state_dict(load_network(osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch))))

    def load_test_checkpoints(self, epoch):
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            if module_name == 'ema_model':
                module = self.modules[i]
                module.load_state_dict(load_network(osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch))))

    def record_metrics(self, epoch, psnr, ssim, rmse):
        new_row = pd.DataFrame([[epoch, psnr, ssim, rmse]], columns=['epoch', 'PSNR', 'SSIM', 'RMSE'])
        self.metrics_df = pd.concat([self.metrics_df, new_row], ignore_index=True)
        self.metrics_df.to_csv(osp.join(self.metrics_save_dir, 'metrics.csv'), index=False)  # 保存为 CSV 文件

    def msg(self, stats, step):
        output_str = '[{}] {:05d}, '.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), step)

        for i in range(len(stats)):
            if isinstance(stats, (list, tuple)):
                var = stats[i]
                var_name = get_varname(stats[i])
            elif isinstance(stats, dict):
                var_name, var = list(stats.items())[i]
            else:
                raise NotImplementedError
            if isinstance(var, torch.Tensor):
                var = var.detach().mean()
                var = reduce_tensor(var)
                var = var.item()
            output_str += '{} {:2.5f}, '.format(var_name, var)

        if self.local_rank == 0:
            print(output_str)

    def train_msg(self, stats, step):
        output_str = '[{}] {:05d}, '.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), step)

        for i in range(len(stats)):
            if isinstance(stats, (list, tuple)):
                var = stats[i]
                var_name = get_varname(stats[i])
            elif isinstance(stats, dict):
                var_name, var = list(stats.items())[i]
            else:
                raise NotImplementedError
            if isinstance(var, torch.Tensor):
                var = var.detach().mean()
                var = reduce_tensor(var)
                var = var.item()
            output_str += '{} {:2.5f}, '.format(var_name, var)

        if self.local_rank == 0:
            print(output_str)
            # 提取 loss 值并保存
            if isinstance(stats, dict):
                tra_loss = stats['loss']  # 如果 stats 是字典，直接取 loss
            elif isinstance(stats, (list, tuple)):
                tra_loss = stats[0]  # 假设 stats[0] 是 loss
            else:
                raise ValueError("Unsupported type for stats")

            self.train_losses.append(tra_loss)

    def val_msg(self, stats, step):
        output_str = '[{}] {:05d}, '.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), step)

        for i in range(len(stats)):
            if isinstance(stats, (list, tuple)):
                var = stats[i]
                var_name = get_varname(stats[i])
            elif isinstance(stats, dict):
                var_name, var = list(stats.items())[i]
            else:
                raise NotImplementedError
            if isinstance(var, torch.Tensor):
                var = var.detach().mean()
                var = reduce_tensor(var)
                var = var.item()
            output_str += '{} {:2.5f}, '.format(var_name, var)

        if self.local_rank == 0:
            print(output_str)
            # 提取 loss 值并保存
            if isinstance(stats, dict):
                val_loss = stats['loss']  # 如果 stats 是字典，直接取 loss
            elif isinstance(stats, (list, tuple)):
                val_loss = stats[0]  # 假设 stats[0] 是 loss
            else:
                raise ValueError("Unsupported type for stats")

            self.val_losses.append(val_loss)


    def save_tifimage(self, grid_img, n_iter, sample_type):
        # 构造保存路径
        save_path = osp.join(
            self.tif_save_dir,
            '{}_{}_{}.tif'.format(n_iter, self.local_rank, sample_type)
        )

        # 确保保存路径的目录存在
        save_dir = os.path.dirname(save_path)  # 获取保存路径的目录部分
        os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在，则创建

        # 保存图像网格到指定路径
        save_image(grid_img, save_path, nrow=1)

        # 输出保存的文件路径以供调试
        print(f"Image saved at: {save_path}")

    def save_tif(self, img, n_iter, sample_type, ref_img=None, min_val=None, max_val=None):
        """
        保存 3D 图像为 TIFF 格式，支持根据参考图像或自定义范围进行反归一化。

        :param img: 要保存的图像，张量或 numpy 数组，形状为 (depth, height, width) 或 (1, 1, depth, height, width)。
        :param n_iter: 当前训练步骤，用于文件命名。
        :param sample_type: 样本类型，用于文件命名。
        :param ref_img: 参考图像，用于恢复原始物理范围（优先）。
        :param min_val: 指定的最小值（次优先）。
        :param max_val: 指定的最大值（次优先）。
        """
        import numpy as np
        import os
        import os.path as osp
        import tifffile as tiff
        import torch

        # 转为 numpy
        img = img.cpu().numpy() if isinstance(img, torch.Tensor) else img
        print(f"save_tif: original shape = {img.shape}, range = ({img.min():.6f}, {img.max():.6f})")

        # 若为 5D (1, 1, D, H, W)，转换成 (D, H, W)
        if img.ndim == 5:
            img = img[0, 0]
        elif img.ndim == 4:
            img = img[0]
        elif img.ndim != 3:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        # 获取参考图像范围
        if ref_img is not None:
            ref_img = ref_img.cpu().numpy() if isinstance(ref_img, torch.Tensor) else ref_img
            if ref_img.ndim == 5:
                ref_img = ref_img[0, 0]
            elif ref_img.ndim == 4:
                ref_img = ref_img[0]
            elif ref_img.ndim != 3:
                raise ValueError(f"Unexpected ref_img shape: {ref_img.shape}")
            MIN_B, MAX_B = np.min(ref_img), np.max(ref_img)
            if MAX_B <= MIN_B or MAX_B - MIN_B < 1e-6:
                print(f"Warning: Invalid ref_img range ({MIN_B:.6f}, {MAX_B:.6f}), fallback to [0, 255]")
                MIN_B, MAX_B = 0, 255
        elif min_val is not None and max_val is not None:
            MIN_B, MAX_B = min_val, max_val
            if MAX_B <= MIN_B or MAX_B - MIN_B < 1e-6:
                print(f"Warning: Invalid min/max range ({MIN_B:.6f}, {MAX_B:.6f}), fallback to [0, 255]")
                MIN_B, MAX_B = 0, 255
        else:
            print("No ref_img or range provided, using [0, 255]")
            MIN_B, MAX_B = 0, 255

        print(f"save_tif: scaling to ({MIN_B:.6f}, {MAX_B:.6f})")

        # 恢复原始像素范围
        img = img * (MAX_B - MIN_B) + MIN_B
        img = np.clip(img, 0, 255)
        print(f"save_tif: after scaling = ({img.min():.6f}, {img.max():.6f})")

        img = img.astype(np.uint16)
        print(f"save_tif: converted to uint16: ({img.min()}, {img.max()})")

        # 保存路径
        save_path = osp.join(self.tif_save_dir, f'{n_iter}_{self.local_rank}_{sample_type}.tif')
        save_dir = osp.dirname(save_path)
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")

        # 保存多页 TIFF
        tiff.imwrite(save_path, img, photometric='minisblack')
        print(f"3D TIFF image saved at {save_path}")

    def save_best_model(self, epoch):
        if self.local_rank != 0:
            # 如果不是主进程，则不保存检查点
            return

        # 保存最佳模型
        for i in range(len(self.modules)):
            module_name = self.module_names[i]  # 获取模块的名称
            module = self.modules[i]  # 获取模块
            # 保存模块的状态字典到指定路径
            torch.save(module.state_dict(), osp.join(self.models_save_dir, '{}-best'.format(module_name)))
        print(f"Best model saved at epoch {epoch}")

    def plot_metrics(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(16, 6))  # 调整尺寸以适应单个图

        # 绘制 Loss 曲线
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        save_path = osp.join(self.metrics_save_dir, 'loss_curve.png')  # 保存文件名为 loss_curve.png
        plt.savefig(save_path)
        plt.close()
        print(f"Loss curve saved at {save_path}")


