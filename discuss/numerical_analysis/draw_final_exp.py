import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 使用 LaTeX 风格渲染，使图表更具论文风格
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 12
})

# 数据输入
data_tmp = [[16.9928, 14.0573], [28.0713, 25.0611], [47.2784, 46.5793],
            [63.3984, 59.7281], [3.9816, 3.8540], [8.7501, 7.8832],
            [160.3655, 131.4325], [57.3999, 54.1212], [196.8050, 195.3460],
            [6.7924, 6.5384]]
data = {
    "Circuit": [
        'c1355', 'c1908', 'c2670', 'c3540', 'c432', 'c499', 'c5315', 'c6288',
        'c7552', 'c880'
    ],
    "Original (s)": [a[0] for a in data_tmp],
    "Optimized (s)": [a[1] for a in data_tmp]
}

# 转换为 DataFrame
df = pd.DataFrame(data)
df["Speedup"] = df["Original (s)"] / df["Optimized (s)"]

# 绘图准备
x = np.arange(len(df["Circuit"]))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 4))
bar1 = ax.bar(x - width / 2, df["Original (s)"], width, label='Original')
bar2 = ax.bar(x + width / 2, df["Optimized (s)"], width, label='Reduced')

# 标注加速比
for i in range(len(x)):
    speedup = df["Speedup"][i]
    if i == 8:
        ax.text(x[i] + width / 4 * 3,
                df["Optimized (s)"][i] + 1,
                f"{speedup:.1f}×",
                ha='center',
                va='bottom',
                fontsize=12,
                rotation=45)
    else:
        ax.text(x[i] + width / 4 * 3,
                df["Optimized (s)"][i] + 1,
                f"{speedup:.3f}×",
                ha='center',
                va='bottom',
                fontsize=12,
                rotation=45)

# 图表细节
ax.set_ylim(1, 220)
ax.set_ylabel("Runtime (s)")
ax.set_xlabel("Circuit")
ax.set_title("Comparison of Original and Reduced Runtime per Circuit")
ax.set_xticks(x)
ax.set_xticklabels(df["Circuit"])
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5, axis='y')

plt.tight_layout()
plt.savefig("runtime_comparison.pdf", bbox_inches="tight", dpi=300)
plt.show()
