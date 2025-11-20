import os

def delete_original_files(folder_path, original_indices):
    for idx in original_indices:
        file_id = f"{idx:02d}"
        filename = f"P{file_id}-corrsta-36size.tif"
        file_path = os.path.join(folder_path, filename)

        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✅ 已删除: {filename}")
        else:
            print(f"⚠️ 未找到: {filename}，跳过...")

# 参数设置
folder_path = r"E:\machine_learning\数据\16cgspect_ground3d"
original_file_indices = [1, 2, 4, 5, 11, 12, 18, 19, 22, 25]

# 调用删除函数
delete_original_files(folder_path, original_file_indices)
