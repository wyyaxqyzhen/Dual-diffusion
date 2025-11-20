import os
import numpy as np
import pydicom
from PIL import Image
from glob import glob
from collections import defaultdict


def process_ima_files(input_folder, output_folder):
    # 创建输出文件夹，如果不存在则创建
    os.makedirs(output_folder, exist_ok=True)

    # 读取所有 IMA 文件
    ima_files = glob(os.path.join(input_folder, "*.IMA"))

    # 用字典存储数据，key为 (epoch, patient_id)，value 为层面的排序和图像数据
    data_dict = defaultdict(lambda: defaultdict(list))

    # 遍历所有 IMA 文件，解析文件名并读取图像数据
    for ima_file in ima_files:
        # 获取文件名
        filename = os.path.basename(ima_file)
        parts = filename.split('_')

        # 解析文件名信息
        epoch = parts[0]  # 第一个部分是 epoch 次数
        patient_id = parts[-2]  # 倒数第二个部分是患者ID（P01, P02等）
        slice_index = int(parts[-1].split('.')[0])  # 最后一部分是层面编号（002, 061等）

        # 读取 IMA 文件中的图像数据
        dicom_data = pydicom.dcmread(ima_file, force=True)
        # 如果没有 TransferSyntaxUID，手动设置为 Little Endian 的默认值
        if not hasattr(dicom_data.file_meta, 'TransferSyntaxUID'):
            dicom_data.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        img_array = dicom_data.pixel_array  # 获取图像数据

        # 存储图像数据
        data_dict[epoch][patient_id].append((slice_index, img_array))

    # 遍历每个 epoch，整合重建 tif 文件
    for epoch, patient_data in data_dict.items():
        # 为每个 epoch 创建一个对应的文件夹
        epoch_folder = os.path.join(output_folder, f"epoch_{epoch}")
        os.makedirs(epoch_folder, exist_ok=True)

        # 遍历每个患者的数据
        for patient_id, slices in patient_data.items():
            # 按层面编号排序
            slices_sorted = sorted(slices, key=lambda x: x[0])

            # 提取所有层面的图像数据并合成为多帧 TIF
            img_stack = [Image.fromarray(slice_data) for _, slice_data in slices_sorted]

            # 构建多帧的 TIF 文件
            tif_filename = os.path.join(epoch_folder, f"{patient_id}.tif")
            img_stack[0].save(tif_filename, save_all=True, append_images=img_stack[1:])
            print(f"Saved {tif_filename}")



