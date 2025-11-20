import numpy as np
import pandas as pd
import tifffile
import os  # 用于创建输出目录

def extract_pixel_values_from_coords(tif_path, coord_excel_path, output_csv_path):
    # 读取 3D 图像
    volume = tifffile.imread(tif_path)  # shape: (depth, height, width)
    print(f"Volume shape: {volume.shape}")

    # 读取坐标
    df_coords = pd.read_csv(coord_excel_path)
    if not {'x', 'y', 'z'}.issubset(df_coords.columns):
        raise ValueError("Excel 文件必须包含列名：'x', 'y', 'z'")

    # 提取对应像素值
    values = []
    for idx, row in df_coords.iterrows():
        x, y, z = int(row['x']), int(row['y']), int(row['z'])
        if (0 <= z < volume.shape[0]) and (0 <= y < volume.shape[1]) and (0 <= x < volume.shape[2]):
            val = volume[z, y, x]  # 注意索引顺序：[z, y, x]
        else:
            val = np.nan  # 超出范围设置为 NaN
        values.append(val)

    # 添加新列
    df_coords['value'] = values

    # ✅ 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # 保存为 CSV 文件
    df_coords.to_csv(output_csv_path, index=False)
    print(f"✅ 提取完毕，已保存到 {output_csv_path}")

# 示例调用
extract_pixel_values_from_coords(
    tif_path='/hy-tmp/NDM_16cg/P001/16cg/P001-16cg-36size.tif',
    coord_excel_path='/hy-tmp/HED_edge_value.csv',
    output_csv_path='/hy-tmp/edge_value/P001-edge-values.csv'
)
