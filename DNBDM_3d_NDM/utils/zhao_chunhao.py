import re
from pathlib import Path
import shutil

# 目标汇总文件夹（里面已经有若干 NDM_8cg_Pxxx.tif）
score_dir = Path(r"E:\machine_learning\结果分析\chunhao_16cg")

# 两个来源根路径
ndm_root = Path(r"E:\machine_learning\数据\NDM_16cg")  # 例如 E:\...\NDM_8cg\P001\8CG\P001-8cg-36size.tif
bdm_root = Path(r"E:\machine_learning\结果分析\预测图像\new_BDM_ez_16cg_5t\save_tif")  # 例如 ...\BDM_8cg_P001.tif

# 从 Score 目录现有文件名中提取病人 ID（保持原有零填充，如 P001、P015、P432）
pid_pat = re.compile(r"NDM_16cg_(P\d+)\.tif$", re.IGNORECASE)

# 统计
copied_ndm = 0
copied_bdm = 0
miss_ndm = []
miss_bdm = []

# 遍历 Score 目录里已有的 NDM_8cg_Pxxx.tif，拿到 Pxxx 列表
pids = []
for f in score_dir.glob("NDM_16cg_P*.tif"):
    m = pid_pat.match(f.name)
    if m:
        pids.append(m.group(1))  # 形如 'P001'、'P432'

# 去重保持顺序
seen = set()
pids = [x for x in pids if not (x in seen or seen.add(x))]

print(f"在 {score_dir} 里识别到 {len(pids)} 个病人：{pids[:10]}{' ...' if len(pids)>10 else ''}")

for pid in pids:
    # --- NDM 源：E:\...\NDM_8cg\Pxxx\8CG\Pxxx-8cg-36size.tif
    ndm_src = ndm_root / pid / "16cg" / f"{pid}-16cg-36size.tif"
    if ndm_src.exists():
        dst = score_dir / ndm_src.name  # 不重命名
        shutil.copy2(ndm_src, dst)
        copied_ndm += 1
        print(f"[NDM ok] {ndm_src} -> {dst}")
    else:
        miss_ndm.append(str(ndm_src))
        print(f"[NDM missing] {ndm_src}")

    # --- BDM 源：E:\...\new_BDM_ez_8cg_5t\save_tif\BDM_8cg_Pxxx.tif
    bdm_src = bdm_root / f"BDM_16cg_{pid}.tif"
    if bdm_src.exists():
        dst = score_dir / bdm_src.name  # 不重命名
        shutil.copy2(bdm_src, dst)
        copied_bdm += 1
        print(f"[BDM ok] {bdm_src} -> {dst}")
    else:
        miss_bdm.append(str(bdm_src))
        print(f"[BDM missing] {bdm_src}")

print(f"\n完成：NDM 复制 {copied_ndm} 个，BDM 复制 {copied_bdm} 个。")
if miss_ndm:
    print(f"NDM 缺失 {len(miss_ndm)} 个样本，示例：{miss_ndm[:3]}")
if miss_bdm:
    print(f"BDM 缺失 {len(miss_bdm)} 个样本，示例：{miss_bdm[:3]}")
