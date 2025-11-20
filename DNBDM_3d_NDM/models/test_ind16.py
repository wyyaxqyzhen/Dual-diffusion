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

        dose = '16cg'
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
        self.dose = '16cg'

    @torch.no_grad()
    def test(self):
        best_model_path = osp.join(self.logger.models_save_dir, 'ema_model-best')
        try:
            self.ema_model.load_state_dict(torch.load(best_model_path))
            print("Loaded best model for testing.")
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
data_root = '../data_preprocess/gen_data/spect_16cg_npy'
patient_ids = [
    '001', '017', '049', '065', '161', '177', '273', '289', '337', '385', '801', '817', '833', '849', '961', '977', '993', '1009', '1121', '1137', '1153', '1169', '1281', '1297', '1313', '1329', '1441', '1457', '1473', '1489', '1601', '1617', '1633', '1649', '1761', '1777', '1793', '1809', '1921', '1937', '1953', '1969', '2081', '2097', '2113', '2129', '2241', '2257', '2273', '2289',
    '002', '018', '050', '066', '162', '178', '274', '290', '338', '386', '802', '818', '834', '850', '962', '978', '994', '1010', '1122', '1138', '1154', '1170', '1282', '1298', '1314', '1330', '1442', '1458', '1474', '1490', '1602', '1618', '1634', '1650', '1762', '1778', '1794', '1810', '1922', '1938', '1954', '1970', '2082', '2098', '2114', '2130', '2242', '2258', '2274', '2290',
    '003', '019', '051', '067', '163', '179', '275', '291', '339', '387', '803', '819', '835', '851', '963', '979', '995', '1011', '1123', '1139', '1155', '1171', '1283', '1299', '1315', '1331', '1443', '1459', '1475', '1491', '1603', '1619', '1635', '1651', '1763', '1779', '1795', '1811', '1923', '1939', '1955', '1971', '2083', '2099', '2115', '2131', '2243', '2259', '2275', '2291',
    '004', '020', '052', '068', '164', '180', '276', '292', '340', '388', '804', '820', '836', '852', '964', '980', '996', '1012', '1124', '1140', '1156', '1172', '1284', '1300', '1316', '1332', '1444', '1460', '1476', '1492', '1604', '1620', '1636', '1652', '1764', '1780', '1796', '1812', '1924', '1940', '1956', '1972', '2084', '2100', '2116', '2132', '2244', '2260', '2276', '2292',
    '005', '021', '053', '069', '165', '181', '277', '293', '341', '389', '805', '821', '837', '853', '965', '981', '997', '1013', '1125', '1141', '1157', '1173', '1285', '1301', '1317', '1333', '1445', '1461', '1477', '1493', '1605', '1621', '1637', '1653', '1765', '1781', '1797', '1813', '1925', '1941', '1957', '1973', '2085', '2101', '2117', '2133', '2245', '2261', '2277', '2293',
    '006', '022', '054', '070', '166', '182', '278', '294', '342', '390', '806', '822', '838', '854', '966', '982', '998', '1014', '1126', '1142', '1158', '1174', '1286', '1302', '1318', '1334', '1446', '1462', '1478', '1494', '1606', '1622', '1638', '1654', '1766', '1782', '1798', '1814', '1926', '1942', '1958', '1974', '2086', '2102', '2118', '2134', '2246', '2262', '2278', '2294',
    '007', '023', '055', '071', '167', '183', '279', '295', '343', '391', '807', '823', '839', '855', '967', '983', '999', '1015', '1127', '1143', '1159', '1175', '1287', '1303', '1319', '1335', '1447', '1463', '1479', '1495', '1607', '1623', '1639', '1655', '1767', '1783', '1799', '1815', '1927', '1943', '1959', '1975', '2087', '2103', '2119', '2135', '2247', '2263', '2279', '2295',
    '008', '024', '056', '072', '168', '184', '280', '296', '344', '392', '808', '824', '840', '856', '968', '984', '1000', '1016', '1128', '1144', '1160', '1176', '1288', '1304', '1320', '1336', '1448', '1464', '1480', '1496', '1608', '1624', '1640', '1656', '1768', '1784', '1800', '1816', '1928', '1944', '1960', '1976', '2088', '2104', '2120', '2136', '2248', '2264', '2280', '2296',
    '009', '025', '057', '073', '169', '185', '281', '297', '345', '393', '809', '825', '841', '857', '969', '985', '1001', '1017', '1129', '1145', '1161', '1177', '1289', '1305', '1321', '1337', '1449', '1465', '1481', '1497', '1609', '1625', '1641', '1657', '1769', '1785', '1801', '1817', '1929', '1945', '1961', '1977', '2089', '2105', '2121', '2137', '2249', '2265', '2281', '2297',
    '010', '026', '058', '074', '170', '186', '282', '298', '346', '394', '810', '826', '842', '858', '970', '986', '1002', '1018', '1130', '1146', '1162', '1178', '1290', '1306', '1322', '1338', '1450', '1466', '1482', '1498', '1610', '1626', '1642', '1658', '1770', '1786', '1802', '1818', '1930', '1946', '1962', '1978', '2090', '2106', '2122', '2138', '2250', '2266', '2282', '2298',
    '011', '027', '059', '075', '171', '187', '283', '299', '347', '395', '811', '827', '843', '859', '971', '987', '1003', '1019', '1131', '1147', '1163', '1179', '1291', '1307', '1323', '1339', '1451', '1467', '1483', '1499', '1611', '1627', '1643', '1659', '1771', '1787', '1803', '1819', '1931', '1947', '1963', '1979', '2091', '2107', '2123', '2139', '2251', '2267', '2283', '2299',
    '012', '028', '060', '076', '172', '188', '284', '300', '348', '396', '812', '828', '844', '860', '972', '988', '1004', '1020', '1132', '1148', '1164', '1180', '1292', '1308', '1324', '1340', '1452', '1468', '1484', '1500', '1612', '1628', '1644', '1660', '1772', '1788', '1804', '1820', '1932', '1948', '1964', '1980', '2092', '2108', '2124', '2140', '2252', '2268', '2284', '2300',
    '013', '029', '061', '077', '173', '189', '285', '301', '349', '397', '813', '829', '845', '861', '973', '989', '1005', '1021', '1133', '1149', '1165', '1181', '1293', '1309', '1325', '1341', '1453', '1469', '1485', '1501', '1613', '1629', '1645', '1661', '1773', '1789', '1805', '1821', '1933', '1949', '1965', '1981', '2093', '2109', '2125', '2141', '2253', '2269', '2285', '2301',
    '014', '030', '062', '078', '174', '190', '286', '302', '350', '398', '814', '830', '846', '862', '974', '990', '1006', '1022', '1134', '1150', '1166', '1182', '1294', '1310', '1326', '1342', '1454', '1470', '1486', '1502', '1614', '1630', '1646', '1662', '1774', '1790', '1806', '1822', '1934', '1950', '1966', '1982', '2094', '2110', '2126', '2142', '2254', '2270', '2286', '2302',
    '015', '031', '063', '079', '175', '191', '287', '303', '351', '399', '815', '831', '847', '863', '975', '991', '1007', '1023', '1135', '1151', '1167', '1183', '1295', '1311', '1327', '1343', '1455', '1471', '1487', '1503', '1615', '1631', '1647', '1663', '1775', '1791', '1807', '1823', '1935', '1951', '1967', '1983', '2095', '2111', '2127', '2143', '2255', '2271', '2287', '2303',
    '016', '032', '064', '080', '176', '192', '288', '304', '352', '400', '816', '832', '848', '864', '976', '992', '1008', '1024', '1136', '1152', '1168', '1184', '1296', '1312', '1328', '1344', '1456', '1472', '1488', '1504', '1616', '1632', '1648', '1664', '1776', '1792', '1808', '1824', '1936', '1952', '1968', '1984', '2096', '2112', '2128', '2144', '2256', '2272', '2288', '2304'
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
    osp.dirname(osp.dirname(osp.abspath(__file__))), 'output', 'corediff_16cgspect_5t'
))

# 运行测试
model = Model(test_loader, ema_model, logger)
model.test()
model.generate_images()