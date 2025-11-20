import numpy as np
import os
import pandas as pd
from tifffile import imread
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

def pad_to_shape(array, target_shape=(64, 64, 64)):
    """对输入 3D 图像进行深度方向的填充，扩展为目标形状"""
    depth, height, width = array.shape
    target_depth, target_height, target_width = target_shape

    pad_depth = max(0, target_depth - depth)
    pad_depth_before = pad_depth // 2
    pad_depth_after = pad_depth - pad_depth_before

    padded_array = np.pad(array,
                          ((pad_depth_before, pad_depth_after), (0, 0), (0, 0)),
                          mode='constant',
                          constant_values=0)
    return padded_array

def data_show(full_dose_dir, denoised_dir, QA_save_path):
    # **手动建立文件名映射**
    mapping = {
        "data1": "150000_0_test_10_ddim_P01.tif",
        "data2": "150000_0_test_10_ddim_P02.tif",
        "data3": "150000_0_test_10_ddim_P03.tif",
        "data4": "150000_0_test_10_ddim_P04.tif",
        "data5": "150000_0_test_10_ddim_P05.tif",
        "data6": "150000_0_test_10_ddim_P06.tif",
        "data7": "150000_0_test_10_ddim_P07.tif",
        "data8": "150000_0_test_10_ddim_P08.tif",
        "data9": "150000_0_test_10_ddim_P09.tif",
        "data10": "150000_0_test_10_ddim_P10.tif",
    }

    file_list = sorted(os.listdir(full_dose_dir))  # 获取金标准文件
    match_count = 0  # 计数匹配的文件数

    # **存储结果**
    results = []

    for full_dose_file in file_list:
        full_dose_path = os.path.join(full_dose_dir, full_dose_file)

        # **提取金标准文件 ID，例如 "data1"**
        file_id = full_dose_file.split('_')[0]

        # **找到对应的预测文件**
        denoised_file = mapping.get(file_id, None)
        if denoised_file is None:
            print(f"未找到 {file_id} 对应的预测文件，跳过...")
            continue

        denoised_path = os.path.join(denoised_dir, denoised_file)

        # 读取 TIFF 图像
        full_dose_img = imread(full_dose_path)
        denoised_img = imread(denoised_path)

        # **补充金标准数据为 (64, 64, 64)**
        full_dose_img = pad_to_shape(full_dose_img, (64, 64, 64))

        # 归一化处理
        NMSE_scale = np.sum(full_dose_img) / np.sum(denoised_img)
        denoised_img = denoised_img * NMSE_scale

        # 计算误差指标
        ME = np.sum(denoised_img - full_dose_img) / full_dose_img.size
        MAE = np.abs(ME)
        NMSE = mean_squared_error(full_dose_img, denoised_img) / np.mean(full_dose_img**2)
        PSNR = peak_signal_noise_ratio(full_dose_img, denoised_img, data_range=np.max(full_dose_img))
        SSIM = structural_similarity(full_dose_img, denoised_img, data_range=np.max(full_dose_img))

        # **存入结果列表**
        results.append([full_dose_file, ME, MAE, NMSE, PSNR, SSIM])

        print(f"已处理: {full_dose_file} <-> {denoised_file}")
        match_count += 1

    # **创建 DataFrame 并保存为 CSV**
    df = pd.DataFrame(results, columns=["Filename", "ME", "MAE", "NMSE", "PSNR", "SSIM"])
    csv_path = os.path.join(QA_save_path, 'QA_results.csv')
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"✅ 结果已保存至: {csv_path}")

if __name__ == '__main__':
    full_dose_dir = '/hy-tmp/Fulldose'  # 金标准数据文件夹
    denoised_dir = '/hy-tmp/corediff_spect_val_3d/output/corediff_dose10s1s_spect/save_tif'  # 预测数据文件夹
    QA_save_path = '/hy-tmp/calculate_output'  # 结果保存路径

    if not os.path.exists(QA_save_path):
        os.makedirs(QA_save_path)

    data_show(full_dose_dir, denoised_dir, QA_save_path)
