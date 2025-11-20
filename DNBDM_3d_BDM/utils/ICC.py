# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pingouin as pg
from sklearn.metrics import cohen_kappa_score

# 你的评分表路径
FILE = r'/hy-tmp/诊断信心打分-对应.xlsx'

# 1. 读取数据
df = pd.read_excel(FILE)

# 如果有中文列名，就优先用列名；否则用列号
if '分数-1' in df.columns and '分数-2' in df.columns:
    r1 = df['分数-1']
    r2 = df['分数-2']
else:
    # 按截图：第5列=打分-1，第6列=打分-2
    r1 = df.iloc[:, 4]
    r2 = df.iloc[:, 5]

# 只保留两位医生都给了分的病例
mask = r1.notna() & r2.notna()
r1 = r1[mask].reset_index(drop=True)
r2 = r2[mask].reset_index(drop=True)

n = len(r1)
print(f"有效病例数: {n}")

# 2. 构造 ICC 需要的长表：每个病例有两行（R1、R2）
icc_df = pd.DataFrame({
    'targets': np.repeat(range(n), 2),    # 0,0,1,1,2,2,...
    'raters': ['R1', 'R2'] * n,           # R1,R2,R1,R2,...
    'ratings': pd.concat([r1, r2], ignore_index=True)
})

# 3. 计算 ICC(2,2)  —— 在 pingouin 里是 Type = 'ICC2k'
icc_res = pg.intraclass_corr(
    data=icc_df,
    targets='targets',
    raters='raters',
    ratings='ratings'
)

icc22 = icc_res[icc_res['Type'] == 'ICC2k'].iloc[0]
icc_value = icc22['ICC']
ci_low, ci_high = icc22['CI95%']

# 4. 计算 quadratic-weighted kappa
kappa_quadratic = cohen_kappa_score(r1, r2, weights='quadratic')

# 5. 打印结果
print("\n===== Inter-observer Reliability (Diagnostic Confidence) =====")
print(f"ICC(2,2) = {icc_value:.3f} (95% CI {ci_low:.3f} – {ci_high:.3f})")
print(f"Quadratic-weighted kappa = {kappa_quadratic:.3f}")
