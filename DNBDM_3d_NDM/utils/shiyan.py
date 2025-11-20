import os
import shutil

def copy_and_rename_multiple_files(source_dir, target_dir, original_indices, copy_count=8):
    os.makedirs(target_dir, exist_ok=True)

    for original_index in original_indices:
        # 格式化编号为两位
        file_id = f"{original_index:02d}"
        filename = f"P{file_id}-corrsta-36size.tif"
        src_path = os.path.join(source_dir, filename)

        if not os.path.exists(src_path):
            print(f"⚠️ 文件不存在: {src_path}，跳过...")
            continue

        start_index = (original_index - 1) * 8 + 1
        for i in range(copy_count):
            new_index = start_index + i
            new_id = f"P{new_index:03d}"
            new_filename = f"{new_id}-corrsta-36size.tif"
            dst_path = os.path.join(target_dir, new_filename)
            shutil.copy(src_path, dst_path)
            print(f"✅ 复制 {filename} -> {new_filename}")

# 参数设置
source_dir = r"E:\machine_learning\数据\8cgspect_ground"
target_dir = r"E:\machine_learning\数据\8cgspect_ground"

# 提供的编号列表
original_file_indices = [1, 2, 4, 5, 11, 12, 18, 19, 22, 25]

# 调用函数
copy_and_rename_multiple_files(source_dir, target_dir, original_file_indices)
