import os
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_squared_error
from datetime import datetime
from tifffile import imread


def normalize_image(img):
    """将图像归一化到 [0, 1] 范围"""
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max == img_min:  # 避免除零
        return img - img_min
    return (img - img_min) / (img_max - img_min)


def data_show(full_dose_folder, gen_dose_folder, QA_save_path):
    # 患者 ID 列表
    patient_ids = [
    '001', '017', '049', '065', '161', '177', '273', '289', '337', '385',
    '002', '018', '050', '066', '162', '178', '274', '290', '338', '386',
    '003', '019', '051', '067', '163', '179', '275', '291', '339', '387',
    '004', '020', '052', '068', '164', '180', '276', '292', '340', '388',
    '005', '021', '053', '069', '165', '181', '277', '293', '341', '389',
    '006', '022', '054', '070', '166', '182', '278', '294', '342', '390',
    '007', '023', '055', '071', '167', '183', '279', '295', '343', '391',
    '008', '024', '056', '072', '168', '184', '280', '296', '344', '392',
    '009', '025', '057', '073', '169', '185', '281', '297', '345', '393',
    '010', '026', '058', '074', '170', '186', '282', '298', '346', '394',
    '011', '027', '059', '075', '171', '187', '283', '299', '347', '395',
    '012', '028', '060', '076', '172', '188', '284', '300', '348', '396',
    '013', '029', '061', '077', '173', '189', '285', '301', '349', '397',
    '014', '030', '062', '078', '174', '190', '286', '302', '350', '398',
    '015', '031', '063', '079', '175', '191', '287', '303', '351', '399',
    '016', '032', '064', '080', '176', '192', '288', '304', '352', '400']

    # 定义组别（10 组，每组 8 个患者）
    groups = groups = [
    ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016'],  # Group 1
    ['017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032'],  # Group 2
    ['049', '050', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', '061', '062', '063', '064'],  # Group 3
    ['065', '066', '067', '068', '069', '070', '071', '072', '073', '074', '075', '076', '077', '078', '079', '080'],  # Group 4
    ['161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176'],  # Group 5
    ['177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192'],  # Group 6
    ['273', '274', '275', '276', '277', '278', '279', '280', '281', '282', '283', '284', '285', '286', '287', '288'],  # Group 7
    ['289', '290', '291', '292', '293', '294', '295', '296', '297', '298', '299', '300', '301', '302', '303', '304'],  # Group 8
    ['337', '338', '339', '340', '341', '342', '343', '344', '345', '346', '347', '348', '349', '350', '351', '352'],  # Group 9
    ['385', '386', '387', '388', '389', '390', '391', '392', '393', '394', '395', '396', '397', '398', '399', '400'],  # Group 10
]


    # Prepare data to write to Excel
    detailed_results = []  # 存储所有详细结果
    group_results = []    # 存储每组平均结果

    # Process each patient
    for patient_id in patient_ids:
        full_dose_filename = f'P{patient_id}-corrsta-36size.tif'
        gen_dose_filename = f'150000_0_test_16cg_ddim_P{patient_id}.tif'
        full_dose_file = os.path.join(full_dose_folder, full_dose_filename)
        gen_dose_file = os.path.join(gen_dose_folder, gen_dose_filename)

        # Check if files exist
        if not os.path.exists(full_dose_file):
            print(f"Warning: Full-dose file not found for {full_dose_filename}")
            continue
        if not os.path.exists(gen_dose_file):
            print(f"Warning: Generated dose file not found for {gen_dose_filename}")
            continue

        # Read images
        full_dose_img = imread(full_dose_file)
        gen_dose_img = imread(gen_dose_file)

        # Verify image shapes
        if full_dose_img.shape != (36, 36, 36):
            print(f"Skipping {full_dose_filename}: Expected shape (36, 36, 36), got {full_dose_img.shape}")
            continue
        if gen_dose_img.shape != (36, 36, 36):
            print(f"Skipping {gen_dose_filename}: Expected shape (36, 36, 36), got {gen_dose_img.shape}")
            continue

        print(f"Processing pair: {full_dose_filename} vs {gen_dose_filename}")

        # Normalize images for PSNR, SSIM, NMSE
        full_dose_img_norm = normalize_image(full_dose_img)
        gen_dose_img_norm = normalize_image(gen_dose_img)

        # Calculate metrics
        h = np.size(full_dose_img)

        # ME and MAE
        ME = np.sum(gen_dose_img - full_dose_img) / h
        MAE = np.abs(ME)

        # NMSE (using normalized images)
        NMSE = mean_squared_error(full_dose_img_norm.flatten(),
                                 gen_dose_img_norm.flatten()) / np.mean(full_dose_img_norm ** 2)

        # PSNR
        MSE = mean_squared_error(full_dose_img_norm.flatten(), gen_dose_img_norm.flatten())
        PSNR = 10 * np.log10(1.0 / MSE) if MSE != 0 else float('inf')

        # SSIM
        SSIM = structural_similarity(full_dose_img_norm, gen_dose_img_norm,
                                    data_range=1.0, multichannel=False)

        # Store detailed results
        detailed_results.append([full_dose_filename, gen_dose_filename, ME, MAE, NMSE, PSNR, SSIM])

    # Calculate group averages
    for group_idx, group_ids in enumerate(groups, 1):
        group_metrics = {'ME': [], 'MAE': [], 'NMSE': [], 'PSNR': [], 'SSIM': []}
        for patient_id in group_ids:
            # Find metrics for this patient
            for result in detailed_results:
                if f'P{patient_id}-corrsta-36size.tif' in result[0]:
                    group_metrics['ME'].append(result[2])
                    group_metrics['MAE'].append(result[3])
                    group_metrics['NMSE'].append(result[4])
                    group_metrics['PSNR'].append(result[5])
                    group_metrics['SSIM'].append(result[6])

        # Calculate averages if group has data
        if group_metrics['ME']:
            group_results.append([
                f'Group {group_idx}',
                ','.join([f'P{id}' for id in group_ids]),
                np.mean(group_metrics['ME']),
                np.mean(group_metrics['MAE']),
                np.mean(group_metrics['NMSE']),
                np.mean(group_metrics['PSNR']),
                np.mean(group_metrics['SSIM'])
            ])

    # Save results to Excel
    detailed_df = pd.DataFrame(detailed_results,
                              columns=['Full_Dose_Filename', 'Gen_Dose_Filename', 'ME', 'MAE', 'NMSE', 'PSNR', 'SSIM'])
    group_df = pd.DataFrame(group_results,
                           columns=['Group', 'Patient_IDs', 'Avg_ME', 'Avg_MAE', 'Avg_NMSE', 'Avg_PSNR', 'Avg_SSIM'])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(QA_save_path, f'QA_results_36size_3d_{timestamp}.xlsx')

    # Write to Excel with two sheets
    with pd.ExcelWriter(result_file, engine='openpyxl') as writer:
        detailed_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
        group_df.to_excel(writer, sheet_name='Group_Average_Results', index=False)

    print(f"Results saved to {result_file}")


if __name__ == '__main__':
    # Define paths
    full_dose_folder = '/hy-tmp/16cgspect_ground3d'  # 全剂量图像文件夹
    gen_dose_folder = '../output/corediff_16cgspect_10t/save_tif'  # 降噪图像文件夹
    QA_save_path = '../output/corediff_16cgspect_100t_duibi'  # 结果保存路径

    # Ensure result directory exists
    os.makedirs(QA_save_path, exist_ok=True)

    # Run processing
    data_show(full_dose_folder, gen_dose_folder, QA_save_path)

    # 验证图像范围（可选）
    # import tifffile
    # full_dose_img = tifffile.imread(os.path.join(full_dose_folder, 'P001-corrsta-36size.tif'))
    # print(f"Full dose range: {full_dose_img.min()} to {full_dose_img.max()}")