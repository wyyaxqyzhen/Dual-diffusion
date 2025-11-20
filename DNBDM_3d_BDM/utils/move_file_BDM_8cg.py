import os
import shutil
from glob import glob
import re

def replace_patient_images(new_images_dir, original_images_root):
    """
    将新生成的图像替换原始病人的图像文件

    参数:
        new_images_dir (str): 存放新图像的目录，例如 /corediff_BDM_8cg_5t/
        original_images_root (str): 原始图像的根路径，例如 /BDM_8cg/
    """
    # 匹配所有新图像
    tif_paths = glob(os.path.join(new_images_dir, "150000_0_test_8cg_ddim_P*.tif"))
    print(f"共找到 {len(tif_paths)} 个待替换的图像")

    for tif_path in tif_paths:
        filename = os.path.basename(tif_path)

        # 用正则提取 P001、P002、...P400 等病人 ID
        match = re.search(r'P\d{3}', filename)
        if not match:
            print(f"[!] 未找到病人ID，跳过文件: {filename}")
            continue
        patient_id = match.group()  # 例如 P400

        # 构造原图像的完整路径
        original_path = os.path.join(original_images_root, patient_id, "sta", f"{patient_id}-corrsta-36size.tif")

        if os.path.exists(original_path):
            # 替换文件
            shutil.copy2(tif_path, original_path)
            print(f"[✓] 替换: {original_path}")
        else:
            print(f"[!] 找不到原图像: {original_path}，跳过")

# 示例调用
if __name__ == "__main__":
    new_images_folder = "/hy-tmp/corediff_spect_val_3d_NDM/output/corediff_8cgspect_5t/save_tif"
    old_images_root = "/hy-tmp/BDM_8cg"
    replace_patient_images(new_images_folder, old_images_root)
