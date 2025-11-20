import re
from pathlib import Path
import shutil

# 源与目标根路径
src_dir = Path(r"E:\machine_learning\结果分析\预测图像\new_NDM_16cg_5t\save_tif")
dst_root = Path(r"E:\machine_learning\数据\BDM_16cg")

# 匹配 NDM_8cg_P001.tif / NDM_8cg_P432.tif 等
pat = re.compile(r"^NDM_16cg_(P(?P<pid>\d{1,4}))\.tif$", re.IGNORECASE)

count = 0
for src in src_dir.glob("NDM_16cg_P*.tif"):
    m = pat.match(src.name)
    if not m:
        print(f"[skip] {src.name}")
        continue

    pid_raw = m.group("pid")            # 保持原始位数：001、432 等
    folder_id = f"P{pid_raw}"           # 目录名与文件名一致：P001、P432（不补零）

    # 目标目录和文件名
    dst_dir = dst_root / folder_id / "sta"
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst_name = f"P{pid_raw}-corrsta-36size.tif"  # 目标文件名也保持原始位数
    dst = dst_dir / dst_name

    # 复制并覆盖
    shutil.copy2(src, dst)
    print(f"[ok] {src.name}  ->  {dst}")
    count += 1

print(f"完成复制并重命名：{count} 个文件")
