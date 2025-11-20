import pandas as pd
import os

def extract_edge_voxel_values(contour_csv, voxel_csv, output_csv):
    # 读取轮廓坐标
    contour_df = pd.read_csv(contour_csv)

    # 读取全部体素值
    voxel_df = pd.read_csv(voxel_csv)

    # 合并两个表格，按 x, y, z 匹配
    merged_df = pd.merge(contour_df, voxel_df, on=['x', 'y', 'z'], how='inner')

    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # 保存边缘像素值
    merged_df.to_csv(output_csv, index=False)
    print(f"Saved {len(merged_df)} edge voxel values to {output_csv}")

# 示例调用
extract_edge_voxel_values(
    contour_csv='/hy-tmp/position/P001-contours.csv',
    voxel_csv='/hy-tmp/value/P001-voxel-values.csv',
    output_csv='/hy-tmp/edge_value/P001-edge-values.csv'
)
