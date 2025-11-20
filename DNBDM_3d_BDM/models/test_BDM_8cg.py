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
            base_input.extend(target_paths)  # 添加所有匹配路径

        dose = '8cg'
        # 加载输入图像路径
        for id in self.patient_ids:
            id_str = str(id).zfill(2)
            input_paths = sorted(glob(osp.join(self.data_root, f'P{id_str}_{dose}_tif.npy')))
            print(f"Searching for input files: {osp.join(self.data_root, f'P{id_str}_{dose}_tif.npy')}")
            print(f"Found input files: {input_paths}")

            if not input_paths:
                print(f"No input files found for patient ID {id_str}, skipping.")
                continue
            base_target.extend(input_paths)

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

    # def normalize_(self, img, MIN_B=0, MAX_B=65535):
    #     img[img < MIN_B] = MIN_B
    #     img[img > MAX_B] = MAX_B
    #     return (img - MIN_B) / (MAX_B - MIN_B)
    def normalize_(self, img):
        MIN_B = np.min(img)  # 动态获取最小值
        MAX_B = np.max(img)  # 动态获取最大值
        img = np.clip(img, MIN_B, MAX_B)  # 限制在最小最大之间
        return (img - MIN_B) / (MAX_B - MIN_B + 1e-8)  # 防止除以 0


class Model:
    def __init__(self, test_loader, ema_model, logger):
        self.test_loader = test_loader
        self.ema_model = ema_model

        self.logger = logger
        self.T = 5  # 采样步数
        self.sampling_routine = "ddim"  # 使用ddim采样
        self.start_adjust_iter = 1  # 采样调整起始步数
        self.dose = '8cg'

    @torch.no_grad()
    def test(self):
        best_model_path = osp.join(self.logger.models_save_dir, 'ema_model-best')
        try:
            self.ema_model.load_state_dict(torch.load(best_model_path))
            print("Loaded best model for testing from:", best_model_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file {best_model_path} not found.")

        self.ema_model.eval()
        n_iter = 150000
        psnr, ssim, rmse = 0., 0., 0.
        valid_batches = 0

        for low_dose, full_dose in tqdm.tqdm(self.test_loader, desc='test'):
            low_dose, full_dose = low_dose.cuda(), full_dose.cuda()

            if len(low_dose.shape) == 4:
                low_dose = low_dose.unsqueeze(1)
            if len(full_dose.shape) == 4:
                full_dose = full_dose.unsqueeze(1)

            gen_full_dose, _, _ = self.ema_model.sample(
                batch_size=low_dose.shape[0],
                img=low_dose,
                t=self.T,
                sampling_routine=self.sampling_routine,
                start_adjust_iter=self.start_adjust_iter,
            )

            full_dose_norm = self.normalize(full_dose)
            gen_full_dose_norm = self.normalize(gen_full_dose)

            data_range = np.max(full_dose_norm) - np.min(full_dose_norm)
            psnr_score, ssim_score, rmse_score = compute_measure(full_dose_norm, gen_full_dose_norm, data_range)
            psnr += psnr_score
            ssim += ssim_score
            rmse += rmse_score
            valid_batches += 1

        if valid_batches > 0:
            psnr /= valid_batches
            ssim /= valid_batches
            rmse /= valid_batches

        self.logger.msg([psnr, ssim, rmse], n_iter)
        self.logger.record_metrics(n_iter, psnr, ssim, rmse)

    def normalize(self, img):
        img_np = img.cpu().numpy() if isinstance(img, torch.Tensor) else img
        MIN_B = np.min(img_np)
        MAX_B = np.max(img_np)
        if MAX_B <= MIN_B:
            return img_np
        img_np = (img_np - MIN_B) / (MAX_B - MIN_B)
        return img_np

    def transfer_calculate_window(self, img, ref_img=None, for_display=True, cut_min=None, cut_max=None):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if isinstance(ref_img, torch.Tensor):
            ref_img = ref_img.cpu().numpy()

        if ref_img is not None:
            MIN_B = np.min(ref_img)
            MAX_B = np.max(ref_img)
            if MAX_B <= MIN_B:
                print(f"Warning: Invalid ref_img range ({MIN_B:.6f}, {MAX_B:.6f}), using fallback [0, 1]")
                MIN_B, MAX_B = 0.0, 1.0
        else:
            MIN_B = np.min(img)
            MAX_B = np.max(img)
            if MAX_B <= MIN_B:
                MIN_B, MAX_B = 0.0, 1.0

        img = img * (MAX_B - MIN_B) + MIN_B
        img = np.clip(img, cut_min if cut_min is not None else MIN_B,
                      cut_max if cut_max is not None else MAX_B)

        if for_display:
            if cut_max == cut_min:
                img = np.zeros_like(img)
            else:
                img = 255 * (img - (cut_min if cut_min is not None else MIN_B)) / \
                      ((cut_max if cut_max is not None else MAX_B) - (cut_min if cut_min is not None else MIN_B))
            return img.astype(np.uint8)
        else:
            return img

    def transfer_display_window(self, img, ref_img=None, cut_min=None, cut_max=None):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if isinstance(ref_img, torch.Tensor):
            ref_img = ref_img.cpu().numpy()

        if ref_img is not None:
            MIN_B = np.min(ref_img)
            MAX_B = np.max(ref_img)
            if MAX_B <= MIN_B:
                print(f"Warning: Invalid ref_img range ({MIN_B:.6f}, {MAX_B:.6f}), using fallback [0, 1]")
                MIN_B, MAX_B = 0.0, 1.0
        else:
            MIN_B = np.min(img)
            MAX_B = np.max(img)
            if MAX_B <= MIN_B:
                MIN_B, MAX_B = 0.0, 1.0

        img = img * (MAX_B - MIN_B) + MIN_B
        img = np.clip(img, cut_min if cut_min is not None else MIN_B,
                      cut_max if cut_max is not None else MAX_B)

        if cut_max == cut_min:
            img = np.zeros_like(img)
        else:
            img = (img - (cut_min if cut_min is not None else MIN_B)) / \
                  ((cut_max if cut_max is not None else MAX_B) - (cut_min if cut_min is not None else MIN_B))
            img = 255 * img
        return img.astype(np.uint8)

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
data_root = '../data_preprocess/gen_data/spect_8cg_npy'
patient_ids =  [
    '001', '009', '025', '033', '081', '089', '137', '145', '169', '193', '401', '409', '417', '425', '481', '489', '497', '505', '561', '569', '577', '585', '641', '649', '657', '665', '721', '729', '737', '745', '801', '809', '817', '825', '881', '889', '897', '905', '961', '969', '977', '985', '1041', '1049', '1057', '1065', '1121', '1129', '1137', '1145',
    '002', '010', '026', '034', '082', '090', '138', '146', '170', '194', '402', '410', '418', '426', '482', '490', '498', '506', '562', '570', '578', '586', '642', '650', '658', '666', '722', '730', '738', '746', '802', '810', '818', '826', '882', '890', '898', '906', '962', '970', '978', '986', '1042', '1050', '1058', '1066', '1122', '1130', '1138', '1146',
    '003', '011', '027', '035', '083', '091', '139', '147', '171', '195', '403', '411', '419', '427', '483', '491', '499', '507', '563', '571', '579', '587', '643', '651', '659', '667', '723', '731', '739', '747', '803', '811', '819', '827', '883', '891', '899', '907', '963', '971', '979', '987', '1043', '1051', '1059', '1067', '1123', '1131', '1139', '1147',
    '004', '012', '028', '036', '084', '092', '140', '148', '172', '196', '404', '412', '420', '428', '484', '492', '500', '508', '564', '572', '580', '588', '644', '652', '660', '668', '724', '732', '740', '748', '804', '812', '820', '828', '884', '892', '900', '908', '964', '972', '980', '988', '1044', '1052', '1060', '1068', '1124', '1132', '1140', '1148',
    '005', '013', '029', '037', '085', '093', '141', '149', '173', '197', '405', '413', '421', '429', '485', '493', '501', '509', '565', '573', '581', '589', '645', '653', '661', '669', '725', '733', '741', '749', '805', '813', '821', '829', '885', '893', '901', '909', '965', '973', '981', '989', '1045', '1053', '1061', '1069', '1125', '1133', '1141', '1149',
    '006', '014', '030', '038', '086', '094', '142', '150', '174', '198', '406', '414', '422', '430', '486', '494', '502', '510', '566', '574', '582', '590', '646', '654', '662', '670', '726', '734', '742', '750', '806', '814', '822', '830', '886', '894', '902', '910', '966', '974', '982', '990', '1046', '1054', '1062', '1070', '1126', '1134', '1142', '1150',
    '007', '015', '031', '039', '087', '095', '143', '151', '175', '199', '407', '415', '423', '431', '487', '495', '503', '511', '567', '575', '583', '591', '647', '655', '663', '671', '727', '735', '743', '751', '807', '815', '823', '831', '887', '895', '903', '911', '967', '975', '983', '991', '1047', '1055', '1063', '1071', '1127', '1135', '1143', '1151',
    '008', '016', '032', '040', '088', '096', '144', '152', '176', '200', '408', '416', '424', '432', '488', '496', '504', '512', '568', '576', '584', '592', '648', '656', '664', '672', '728', '736', '744', '752', '808', '816', '824', '832', '888', '896', '904', '912', '968', '976', '984', '992', '1048', '1056', '1064', '1072', '1128', '1136', '1144', '1152'
]

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
    image_size=36,
    timesteps=5,

).cuda()
logger = LoggerX(save_root=osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))), 'output', 'corediff_BDM_8cg_5t'
))

# 运行测试
model = Model(test_loader, ema_model, logger)
model.test()
model.generate_images()