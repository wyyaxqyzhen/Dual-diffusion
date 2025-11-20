from pathlib import Path
import shutil

# 源与目标
src_root = Path(r"E:\machine_learning\结果分析\医生打分\test_groundtruth")
dst_root = Path(r"E:\machine_learning\结果分析\医生打分\打分\随机")
dst_root.mkdir(parents=True, exist_ok=True)

# 允许的扩展名（不区分大小写）
exts = {".png"}

def unique_name(dst_dir: Path, name: str) -> Path:
    """在 dst_dir 下生成不重名的文件路径：abc.png -> abc_1.png / abc_2.png ..."""
    base = Path(name).stem
    suf  = Path(name).suffix
    cand = dst_dir / name
    i = 1
    while cand.exists():
        cand = dst_dir / f"{base}_{i}{suf}"
        i += 1
    return cand

count = 0
for p in src_root.rglob("*"):
    if p.is_file() and p.suffix.lower() in exts:
        dst = unique_name(dst_root, p.name)
        shutil.copy2(p, dst)
        count += 1
        # 如需看到来源子文件夹，可打开下面一行：
        # print(f"[ok] {p} -> {dst}")

print(f"完成：共复制 {count} 张 PNG 到 {dst_root}")
