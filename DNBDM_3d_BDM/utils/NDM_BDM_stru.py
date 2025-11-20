from pathlib import Path
import re
import numpy as np
import pandas as pd
import tifffile

# ========= 参数 =========
score_dir = Path(r"/hy-tmp/D-DDPM_y_16cg")   # NDM/BDM/Excel 都在这里
out_dir   = Path(r"/hy-tmp/D-DDPM_16cg")      # 融合结果输出目录
alpha     = 0.395                         # 融合权重
# ========================

out_dir.mkdir(parents=True, exist_ok=True)

# NDM 文件：NDM_8cg_P001.tif、NDM_8cg_P0432.tif ...
pat_ndm = re.compile(r"^NDM_16cg_(P\d+)\.tif$", re.IGNORECASE)
ndm_files = sorted(score_dir.glob("NDM_16cg_P*.tif"))
if not ndm_files:
    print(f"在 {score_dir} 未找到 NDM_16cg_Pxxx.tif")
    raise SystemExit(0)

done = skipped = 0
for ndm_path in ndm_files:
    m = pat_ndm.match(ndm_path.name)
    if not m:
        skipped += 1
        continue

    pid = m.group(1)  # 例如 P001

    # === 改这里：使用 BDM 图 ===
    bdm_path = score_dir / f"BDM_16cg_{pid}.tif"
    if not bdm_path.exists():
        print(f"[skip] {pid}: 缺少 {bdm_path.name}")
        skipped += 1
        continue

    # Excel：p001.xlsx / P001.xlsx / p001-xxx.xlsx 都尝试
    xlsx_path = score_dir / f"{pid.lower()}.xlsx"
    if not xlsx_path.exists():
        cands = list(score_dir.glob(f"{pid}*.xlsx")) + list(score_dir.glob(f"{pid.lower()}*.xlsx"))
        if cands:
            xlsx_path = cands[0]
        else:
            print(f"[skip] {pid}: 未找到 Excel")
            skipped += 1
            continue

    ndm = tifffile.imread(str(ndm_path)).astype(np.float32)
    bdm = tifffile.imread(str(bdm_path)).astype(np.float32)
    if ndm.shape != bdm.shape:
        print(f"[skip] {pid}: 尺寸不一致 {ndm.shape} vs {bdm.shape}")
        skipped += 1
        continue

    D, H, W = ndm.shape

    # 读取 Excel（忽略 z；将每个 (x,y) 视作所有 z 层的排除点）
    df = pd.read_excel(xlsx_path)
    cols = {c.lower(): c for c in df.columns}
    if not {"x", "y"} <= set(cols):
        print(f"[skip] {pid}: Excel 缺少 x/y 列")
        skipped += 1
        continue
    x_col, y_col = cols["x"], cols["y"]

    # 取有效 (x,y)，越界丢弃；去重
    xs = df[x_col].astype("Int64", errors="ignore")
    ys = df[y_col].astype("Int64", errors="ignore")
    xy = []
    for x, y in zip(xs, ys):
        try:
            xi, yi = int(x), int(y)
        except Exception:
            continue
        if 0 <= xi < W and 0 <= yi < H:
            xy.append((xi, yi))
    xy = sorted(set(xy))

    # 构建“排除掩膜”：这些 (x,y) 在所有 z 层都不融合
    exclude = np.zeros((D, H, W), dtype=bool)
    if xy:
        xx = np.array([p[0] for p in xy])
        yy = np.array([p[1] for p in xy])
        exclude[:, yy[:, None], xx[:, None]] = True  # 广播到所有 z

    include = ~exclude  # 其他位置参与融合

    # 融合：只在 include=True 的位置做 (1-α)*NDM + α*BDM
    fused = ndm.copy()
    fused[include] = (1.0 - alpha) * ndm[include] + alpha * bdm[include]

    out_path = out_dir / f"D-DDPM_16cg_{pid}.tif"
    tifffile.imwrite(str(out_path), fused.astype(np.float32))
    print(f"[ok] {pid}: 全切片按 Excel (x,y) 排除后融合 -> {out_path.name}  shape={fused.shape}")
    done += 1

print(f"\n完成：成功 {done} 个，跳过 {skipped} 个。输出目录：{out_dir}")
