import pandas as pd
import tifffile
import numpy as np
import os

def fuse_voxel_values_with_edge_values(tif_path, edge_csv, output_tif, alpha=0.5):
    # 读取原始图像
    volume = tifffile.imread(tif_path).astype(np.float32)  # 确保是float以支持融合计算

    # 读取边缘像素值
    edge_df = pd.read_csv(edge_csv)

    # 逐点融合
    for _, row in edge_df.iterrows():
        x, y, z, edge_value = int(row['x']), int(row['y']), int(row['z']), float(row['value'])
        original_value = volume[z, y, x]
        combined_value = (1 - alpha) * original_value + alpha * edge_value
        volume[z, y, x] = combined_value

    # 创建输出目录
    os.makedirs(os.path.dirname(output_tif), exist_ok=True)

    # 保存融合后的图像（转换回16位或8位可视化格式，如果需要）
    tifffile.imwrite(output_tif, volume.astype(np.float32))  # 可根据需要改为 uint8 / uint16
    print(f"Saved fused volume to {output_tif}")

# 示例调用
fuse_voxel_values_with_edge_values(
    tif_path='/hy-tmp/corediff_spect_val_3d_BDM/output/corediff_BDM_16cg_5t/save_tif/150000_0_test_16cg_ddim_P001.tif',
    edge_csv='/hy-tmp/edge_value/P001-edge-values.csv',
    output_tif='/hy-tmp/edge_image/P001-edge-fused.tif',
    alpha=0.5  # 可根据需要调整融合权重
)
