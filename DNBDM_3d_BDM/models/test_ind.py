import tqdm
import sys
import os

# 将 corediff_spect_val_3d 添加到模块搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.measure import compute_measure  # 导入计算指标的函数
import wandb
import os.path as osp
from glob import glob
import numpy as np
import torch
import torchvision
from utils.loggerx import LoggerX
from corediff.corediff_wrapper import Network, WeightNet
from corediff.diffusion_modules import Diffusion

class SpectDataLoader:
    def __init__(self, data_root, patient_ids):
        self.data_root = data_root
        self.patient_ids = patient_ids
        self.base_input, self.base_target = self.process_data()

    def process_data(self):
        base_input, base_target = [], []

        # 加载目标图像路径
        for id in self.patient_ids:
            id_str = str(id).zfill(2)  # 确保患者 ID 为两位数
            target_paths = sorted(glob(osp.join(self.data_root, f'P{id_str}_target_tif.npy')))
            print(f"Searching for target files: {osp.join(self.data_root, f'P{id_str}_target_tif.npy')}")
            print(f"Found target files: {target_paths}")

            if not target_paths:
                print(f"No target files found for patient ID {id_str}, skipping.")
                continue
            base_target.extend(target_paths)  # 添加所有匹配路径

        dose = 10
        # 加载输入图像路径
        for id in self.patient_ids:
            id_str = str(id).zfill(2)
            input_paths = sorted(glob(osp.join(self.data_root, f'P{id_str}_{dose}_tif.npy')))
            print(f"Searching for input files: {osp.join(self.data_root, f'P{id_str}_{dose}_tif.npy')}")
            print(f"Found input files: {input_paths}")

            if not input_paths:
                print(f"No input files found for patient ID {id_str}, skipping.")
                continue
            base_input.extend(input_paths)

        return base_input, base_target

    def __getitem__(self, index):
        input_path, target_path = self.base_input[index], self.base_target[index]

        # 加载 3D 数据
        input = np.load(input_path).astype(np.float32)  # 假设输入是 3D 的
        target = np.load(target_path).astype(np.float32)  # 假设目标是 3D 的

        # 归一化
        input = self.normalize_(input)
        target = self.normalize_(target)

        return input, target

    def __len__(self):
        return len(self.base_target)

    def normalize_(self, img, MIN_B=0, MAX_B=65535):
        img[img < MIN_B] = MIN_B
        img[img > MAX_B] = MAX_B
        return (img - MIN_B) / (MAX_B - MIN_B)


class Model:
    def __init__(self, test_loader, ema_model, logger):
        self.test_loader = test_loader
        self.ema_model = ema_model
        self.logger = logger
        self.T = 10  # 采样步数
        self.sampling_routine = "ddim"  # 使用ddim采样
        self.start_adjust_iter = 1  # 采样调整起始步数
        self.dose = 10

    @torch.no_grad()
    def test(self):
        best_model_path = osp.join(self.logger.models_save_dir, 'ema_model-best')
        self.ema_model.load_state_dict(torch.load(best_model_path))
        print("Loaded best model for testing.")
        self.ema_model.eval()
        n_iter = 150000  # 测试迭代次数
        psnr, ssim, rmse = 0., 0., 0.

        for low_dose, full_dose in tqdm.tqdm(self.test_loader, desc='test'):
            low_dose, full_dose = low_dose.cuda(), full_dose.cuda()

            # 添加通道维度，确保是 5D 张量
            if len(low_dose.shape) == 4:  # 如果是 [batch_size, depth, height, width]
                low_dose = low_dose.unsqueeze(1)  # 添加通道维度，变为 [batch_size, channels, depth, height, width]
            if len(full_dose.shape) == 4:  # 如果是 [batch_size, depth, height, width]
                full_dose = full_dose.unsqueeze(1)  # 添加通道维度，变为 [batch_size, channels, depth, height, width]

            gen_full_dose, _, _ = self.ema_model.sample(
                batch_size=low_dose.shape[0],
                img=low_dose,
                t=self.T,
                sampling_routine=self.sampling_routine,
                start_adjust_iter=self.start_adjust_iter,
            )

            # 计算PSNR、SSIM和RMSE
            full_dose = self.transfer_calculate_window(full_dose)
            gen_full_dose = self.transfer_calculate_window(gen_full_dose)
            data_range = full_dose.max() - full_dose.min()

            psnr_score, ssim_score, rmse_score = compute_measure(full_dose, gen_full_dose, data_range)
            psnr += psnr_score / len(self.test_loader)
            ssim += ssim_score / len(self.test_loader)
            rmse += rmse_score / len(self.test_loader)

        self.logger.msg([psnr, ssim, rmse], n_iter)
        self.logger.record_metrics(n_iter, psnr, ssim, rmse)

    def transfer_calculate_window(self, img, MIN_B=1, MAX_B=65535, cut_min=1, cut_max=65535):
        img = img * (MAX_B - MIN_B) + MIN_B  # 将归一化图像恢复到物理值范围
        img[img < cut_min] = cut_min  # 裁剪低于 cut_min 的像素
        img[img > cut_max] = cut_max  # 裁剪高于 cut_max 的像素
        img = 255 * (img - cut_min) / (cut_max - cut_min)  # 映射到 [0, 255] 用于显示
        return img

    def transfer_display_window(self, img, MIN_B=1, MAX_B=65535, cut_min=1, cut_max=65535):
        img = img * (MAX_B - MIN_B) + MIN_B  # 将归一化图像恢复到物理值范围 [MIN_B, MAX_B]
        img[img < cut_min] = cut_min  # 将低于 cut_min 的像素值裁剪为 cut_min
        img[img > cut_max] = cut_max  # 将高于 cut_max 的像素值裁剪为 cut_max
        # 将图像值重新映射到 [0, 1] 的范围，用于显示或进一步处理
        img = (img - cut_min) / (cut_max - cut_min)
        return img  # 返回去归一化并且适合显示的图像

    @torch.no_grad()  # 禁用梯度计算，提高推理效率，减少内存消耗
    def generate_images(self):
        # 加载最优模型
        best_model_path = osp.join(self.logger.models_save_dir, 'ema_model-best')
        self.ema_model.load_state_dict(torch.load(best_model_path))
        print("Loaded best model for testing.")
        self.ema_model.eval()

        n_iter = 150000  # 测试迭代次数

        # 获取低剂量和全剂量图像，确保形状为 [batch_size, depth, height, width]
        low_dose, full_dose = forgen_test_images

        # 添加通道维度，确保形状为 [batch_size, 1, depth, height, width]
        low_dose = low_dose.unsqueeze(1)
        full_dose = full_dose.unsqueeze(1)

        # 使用 EMA 模型生成预测的全剂量图像以及中间的重建图像
        gen_full_dose, direct_recons, imstep_imgs = self.ema_model.sample(
            batch_size=low_dose.shape[0],  # 设定 batch 大小
            img=low_dose,  # 输入低剂量图像
            t=self.T,  # 设定扩散过程的时间步数
            sampling_routine=self.sampling_routine,  # 设定采样例程
            n_iter=n_iter,  # 当前迭代步数
            start_adjust_iter=self.start_adjust_iter,  # 调整开始的迭代步数
        )

        # 获取 batch 大小、通道数、深度、高度和宽度
        b, c, d, h, w = low_dose.size()

        for i in range(b):
            # 从输入路径提取文件 ID
            input_file_path = dataset.base_input[i]  # 数据集的输入路径
            base_filename = os.path.basename(input_file_path)  # 提取文件名，例如 'P01_10_000_img.npy'

            # 文件名示例：'P01_10_000_img.npy'
            file_id = base_filename.split('_')[0]  # 提取文件ID，例如 'P01'

            # 将低剂量图像、全剂量图像和生成的全剂量图像堆叠到一个张量中
            fake_imgs = torch.stack([
                low_dose[i, 0],  # 移除通道维度
                full_dose[i, 0],
                gen_full_dose[i, 0],  # **去掉通道维度**
            ])  # 堆叠后的形状为 [3, depth, height, width]

            # 将数据映射到显示范围
            fake_imgs = self.transfer_display_window(fake_imgs)

            # **保存图像时去掉通道维度**
            self.logger.save_tif(
                img=gen_full_dose[i, 0],  # **去掉通道维度，形状变为 (64, 64, 64)**
                n_iter=n_iter,  # 当前迭代步数
                sample_type=f'test_{self.dose}_{self.sampling_routine}_{file_id}'  # 文件命名的类型
            )


# 初始化数据和模型
data_root = '../data_preprocess/gen_data/spect_10s1s_npy'
patient_ids = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
dataset = SpectDataLoader(data_root, patient_ids)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

# 加载全部测试图像用于测试和可视化
test_images = [dataset[i] for i in range(len(dataset))]
low_dose = torch.stack([torch.from_numpy(x[0]) for x in test_images], dim=0).cuda()
full_dose = torch.stack([torch.from_numpy(x[1]) for x in test_images], dim=0).cuda()
# 保存低剂量和全剂量图像
forgen_test_images = (low_dose, full_dose)
# 保存测试数据集
# self.test_dataset = test_dataset

denoise_fn = Network(in_channels=1, context=False)
# 初始化模型和日志记录器
ema_model = Diffusion(
    denoise_fn=denoise_fn,
    image_size=64,
    timesteps=10,

).cuda()
logger = LoggerX(save_root=osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))), 'output', 'corediff_dose10s1s_spect'
))

# 运行测试
model = Model(test_loader, ema_model, logger)
model.test()
model.generate_images()
