from pathlib import Path
import re

# 你的图片目录（16cg）
base_dir = Path(r"E:\machine_learning\结果分析\医生打分\打分\16cg\GAN_16cg")

# 你之前提供的 20 个原始序号
nums_right = [
    73, 74, 81, 82, 83, 84, 91, 92, 93, 94,
    101, 102, 103, 104, 111, 112, 113, 114, 121, 122
]

# === 计算错误生成的目标编号（用 ×8 生成的那批，需要删除）===
wrong_ids = set()
for n in nums_right:
    wrong_ids.add((n - 1) * 8 + 2)
    wrong_ids.add((n - 1) * 8 + 7)

# 用正则从文件名中提取 P编号（支持3或4位，如 P001 或 P1023）
pat_id = re.compile(r"_P(\d{3,4})", re.IGNORECASE)

deleted, skipped = 0, 0
for img in base_dir.glob("*.png"):  # 只删 png，不碰 tif
    m = pat_id.search(img.name)
    if not m:
        skipped += 1
        continue
    pid = int(m.group(1))
    if pid in wrong_ids:
        try:
            img.unlink()  # 删除文件
            print(f"[del] {img.name}")
            deleted += 1
        except Exception as e:
            print(f"[err] 删除失败 {img.name}: {e}")
    else:
        skipped += 1

print(f"\n完成：已删除 {deleted} 个错误图片，跳过 {skipped} 个。目录：{base_dir}")
