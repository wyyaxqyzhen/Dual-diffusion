import os
import shutil
from glob import glob

def replace_patient_images(new_images_dir, original_images_root):
    """
    将新生成的图像替换原始病人的图像文件

    参数:
        new_images_dir (str): 存放新图像的目录，例如 /corediff_BDM_16cg_5t/
        original_images_root (str): 原始图像的根路径，例如 /BDM_16cg/
    """
    # 匹配所有新图像
    tif_paths = glob(os.path.join(new_images_dir, "P*-16cg-36size.tif"))
    print(f"共找到 {len(tif_paths)} 个待替换的图像")

    for tif_path in tif_paths:
        filename = os.path.basename(tif_path)
        # 提取病人ID，例如 P004
        patient_id = filename.split('-')[0]
        # 构造原图像的完整路径
        original_path = os.path.join(original_images_root, patient_id, "sta", f"{patient_id}-corrsta-36size.tif")

        if os.path.exists(original_path):
            # 备份旧文件（可选）
            # backup_path = original_path + ".bak"
            # shutil.copy2(original_path, backup_path)

            # 替换文件
            shutil.copy2(tif_path, original_path)
            print(f"[✓] 替换: {original_path}")
        else:
            print(f"[!] 找不到原图像: {original_path}，跳过")

# 示例调用
if __name__ == "__main__":
    new_images_folder = "/corediff_BDM_16cg_5t"
    old_images_root = "/BDM_16cg"
    replace_patient_images(new_images_folder, old_images_root)
