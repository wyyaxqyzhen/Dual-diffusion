import os
import os.path as osp
import argparse
import numpy as np
from natsort import natsorted
from glob import glob
import tifffile as tiff  # 导入 tifffile 库来读取多帧 .tif 文件

def save_dataset(args):
    if not osp.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    patient_ids = ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10',
                   'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20',
                   'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30',
                   'P31', 'P32', 'P33', 'P34', 'P35', 'P36', 'P37', 'P38', 'P39', 'P40',
                   'P41', 'P42', 'P43', 'P44', 'P45', 'P46', 'P47', 'P48', 'P49', 'P50']

    def pad_to_shape(array, target_shape=(64, 64, 64)):
        """
        对输入 3D 图像进行深度方向的填充，扩展为目标形状。
        """
        depth, height, width = array.shape
        target_depth, target_height, target_width = target_shape

        # 深度方向计算需要填充的大小
        pad_depth = max(0, target_depth - depth)
        pad_depth_before = pad_depth // 2
        pad_depth_after = pad_depth - pad_depth_before

        # 高度和宽度方向不填充
        pad_height = max(0, target_height - height)
        pad_width = max(0, target_width - width)

        padded_array = np.pad(array,
                              ((pad_depth_before, pad_depth_after), (0, 0), (0, 0)),
                              mode='constant',
                              constant_values=0)
        return padded_array

    io = 'target'
    for p_ind, patient_id in enumerate(patient_ids):
        print(f"Processing patient: {patient_id}")
        if p_ind >= 0:
            patient_path = osp.join(args.data_path, patient_id, 'dose_10s')
            data_paths = natsorted(glob(osp.join(patient_path, '*.tif')))
            for data_path in data_paths:
                f = tiff.imread(data_path)  # 读取为 3D 数组
                print(f"Original shape of the file '{data_path}': {f.shape}")
                f = pad_to_shape(f, (64, 64, 64))  # 填充到目标形状
                print(f"Padded shape of the file '{data_path}': {f.shape}")
                f_name = '{}_{}_tif.npy'.format(patient_id, io)
                np.save(osp.join(args.save_path, f_name), f.astype(np.uint16))  # 保存为 .npy 文件

    io = '10'
    for p_ind, patient_id in enumerate(patient_ids):
        print(f"Processing patient: {patient_id}")
        if p_ind >= 0:
            patient_path = osp.join(args.data_path, patient_id, 'dose_1s')
            data_paths = natsorted(glob(osp.join(patient_path, '*.tif')))
            for data_path in data_paths:
                f = tiff.imread(data_path)  # 读取为 3D 数组
                print(f"Original shape of the file '{data_path}': {f.shape}")
                f = pad_to_shape(f, (64, 64, 64))  # 填充到目标形状
                print(f"Padded shape of the file '{data_path}': {f.shape}")
                f_name = '{}_{}_tif.npy'.format(patient_id, io)
                np.save(osp.join(args.save_path, f_name), f.astype(np.uint16))  # 保存为 .npy 文件


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../data_spect3d')   # 数据路径
    parser.add_argument('--save_path', type=str, default='./gen_data/spect_10s1s_npy/')  # 保存路径
    args = parser.parse_args()
    save_dataset(args)

    if not osp.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))
