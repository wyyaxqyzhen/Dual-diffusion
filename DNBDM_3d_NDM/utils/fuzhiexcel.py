import re
from pathlib import Path
import shutil

base_dir = Path(r"E:\machine_learning\结果分析\医生打分\打分\8cg\D-CNN_8cg")

# ===== 右边图片里给的 20 个“原始序号”，顺序要和你希望的配对顺序一致 =====
nums_right = [
    73,74,81,82,83,84,91,92,93,94,101,102,103,104,111,112,113,114,121,122
]

# 收集并排序 40 张源 PNG
pngs = sorted(base_dir.glob("*.png"))
if len(pngs) == 0:
    raise SystemExit(f"在 {base_dir} 未找到 .png 文件")
if len(pngs) % 2 != 0:
    raise SystemExit(f"源 PNG 数量({len(pngs)})不是偶数，无法两两分组。")
if 2 * len(nums_right) != len(pngs):
    raise SystemExit(
        f"编号个数({len(nums_right)})×2 与 PNG 数量({len(pngs)})不一致。\n"
        f"应满足 2 * len(nums_right) == len(pngs)。"
    )

# 用正则拆分文件名：prefix + P### + suffix（不含扩展名）
pat = re.compile(r"^(?P<prefix>.*?_P)(?P<num>\d{3,4})(?P<suffix>.*)\.png$", re.IGNORECASE)

def make_name(prefix: str, pid_int: int, suffix: str, width: int) -> str:
    """保持原格式，仅替换编号，宽度按原文件编号宽度（通常3位）"""
    return f"{prefix}{pid_int:0{width}d}{suffix}.png"

made = skipped = 0
for i, n in enumerate(nums_right):
    # 该序号对应两张源图：第 2i、2i+1 张
    src_a = pngs[2*i]
    src_b = pngs[2*i + 1]

    # 解析文件名，拿到前缀/后缀与编号宽度
    ma = pat.match(src_a.name)
    mb = pat.match(src_b.name)
    if not ma or not mb:
        print(f"[skip] 文件名无法解析：{src_a.name if not ma else ''} {src_b.name if not mb else ''}")
        skipped += 2
        continue

    width_a = len(ma.group("num"))
    width_b = len(mb.group("num"))
    prefix_a, suffix_a = ma.group("prefix"), ma.group("suffix")
    prefix_b, suffix_b = mb.group("prefix"), mb.group("suffix")

    try:
        n = int(n)
    except Exception:
        print(f"[skip] 非法原始序号：{n}")
        skipped += 2
        continue

    # 目标编号： (n-1)*8+2  与  (n-1)*8+7
    id2 = (n - 1) * 8 + 2
    id7 = (n - 1) * 8 + 7

    # 生成目标名（保持各自源文件的编号位数）
    dst_a = base_dir / make_name(prefix_a, id2, suffix_a, width_a)
    dst_b = base_dir / make_name(prefix_b, id7, suffix_b, width_b)

    # 复制（如不想覆盖可加判断：if not dst.exists():）
    shutil.copy2(src_a, dst_a)
    shutil.copy2(src_b, dst_b)
    print(f"[ok] {src_a.name} -> {dst_a.name}   |   {src_b.name} -> {dst_b.name}")
    made += 2

print(f"\n完成：生成 {made} 张，跳过 {skipped} 张。输出目录：{base_dir}")
