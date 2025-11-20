from pathlib import Path
import re
import numpy as np
import tifffile

# ========= 参数 =========
score_dir = Path(r"/hy-tmp/DDMU-1.0_y_8cg")   # NDM/BDM 都在这里
out_dir   = Path(r"/hy-tmp/DDMU-1.0_8cg")      # 融合结果输出目录
alpha     = 1.0                            # 融合权重
# ========================

out_dir.mkdir(parents=True, exist_ok=True)

# 匹配 NDM 文件：NDM_8cg_P001.tif、NDM_8cg_P0432.tif ...
pat_ndm = re.compile(r"^NDM_8cg_(P\d+)\.tif$", re.IGNORECASE)
ndm_files = sorted(score_dir.glob("NDM_8cg_P*.tif"))
if not ndm_files:
    print(f"在 {score_dir} 未找到 NDM_8cg_Pxxx.tif")
    raise SystemExit(0)

done = skipped = 0
for ndm_path in ndm_files:
    m = pat_ndm.match(ndm_path.name)
    if not m:
        skipped += 1
        continue

    pid = m.group(1)  # 例如 P001

    # 找到对应的 BDM
    bdm_path = score_dir / f"BDM_8cg_{pid}.tif"
    if not bdm_path.exists():
        print(f"[skip] {pid}: 缺少 {bdm_path.name}")
        skipped += 1
        continue

    # 读取数据（float32）
    ndm = tifffile.imread(str(ndm_path)).astype(np.float32)
    bdm = tifffile.imread(str(bdm_path)).astype(np.float32)
    if ndm.shape != bdm.shape:
        print(f"[skip] {pid}: 尺寸不一致 {ndm.shape} vs {bdm.shape}")
        skipped += 1
        continue

    # 全体素直接融合
    fused = (1.0 - alpha) * ndm + alpha * bdm

    # 保存
    out_path = out_dir / f"DDMU-1.0_8cg_{pid}.tif"
    tifffile.imwrite(str(out_path), fused.astype(np.float32))
    print(f"[ok] {pid}: 直接融合 -> {out_path.name}  shape={fused.shape}")
    done += 1

print(f"\n完成：成功 {done} 个，跳过 {skipped} 个。输出目录：{out_dir}")
