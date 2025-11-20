from pathlib import Path
import re
import shutil

# === 路径配置 ===
ddmu_dir = Path(r"E:\machine_learning\结果分析\医生打分\8cg-5t-Score")
sta_root = Path(r"E:\machine_learning\数据\NDM_8cg")
out_dir  = Path(r"E:\machine_learning\结果分析\医生打分\8cg-Score-groundtruth")

out_dir.mkdir(parents=True, exist_ok=True)

# 在文件名里找 P编号（大小写均可、位数 1~4）
pid_pat = re.compile(r"(P\d{1,4})", re.IGNORECASE)

copied = skipped = 0
for p in ddmu_dir.iterdir():
    if not p.is_file():
        continue

    m = pid_pat.search(p.name)
    if not m:
        # 这个文件名里没找到病人编号，跳过
        skipped += 1
        continue

    pid_raw = m.group(1)              # 如 P1 / P001 / P0432
    pid_up  = pid_raw.upper()[1:]     # 去掉前缀 P，只保留数字部分并大写处理
    # 目录名可能有不同的零填充方式，挨个尝试
    candidates = [
        sta_root / f"P{pid_up}" / "sta" / f"P{pid_up}-corrsta-36size.tif",
        sta_root / f"P{pid_up.zfill(3)}" / "sta" / f"P{pid_up.zfill(3)}-corrsta-36size.tif",
        sta_root / f"P{pid_up.zfill(4)}" / "sta" / f"P{pid_up.zfill(4)}-corrsta-36size.tif",
    ]

    src_tif = next((c for c in candidates if c.exists()), None)
    if src_tif is None:
        print(f"[skip] 未找到对应 sta 图：{pid_raw}  (尝试路径数={len(candidates)})")
        skipped += 1
        continue

    dst_tif = out_dir / src_tif.name  # 保持文件名 <PID>-corrsta-36size.tif
    try:
        shutil.copy2(src_tif, dst_tif)  # 覆盖式复制
        print(f"[ok] {pid_raw}: {src_tif} -> {dst_tif}")
        copied += 1
    except Exception as e:
        print(f"[err] 复制失败 {pid_raw}: {e}")
        skipped += 1

print(f"\n完成：复制 {copied} 个，跳过 {skipped} 个。输出目录：{out_dir}")
