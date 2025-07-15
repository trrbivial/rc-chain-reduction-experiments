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
data = {
    "Circuit": [
        'c1355', 'c1908', 'c2670', 'c3540', 'c432', 'c499', 'c5315', 'c6288',
        'c7552', 'c880'
    ],
    "Original (ms)":
    [13074, 18569, 28971, 43406, 6066, 12112, 66139, 58812, 58891, 9928],
    "Optimized (ms)":
    [11418, 16639, 23723, 41746, 5214, 11572, 59420, 56092, 58576, 8678],
}

# 转换为 DataFrame
df = pd.DataFrame(data)
df["Speedup"] = df["Original (ms)"] / df["Optimized (ms)"]

# 绘图准备
x = np.arange(len(df["Circuit"]))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 4))
bar1 = ax.bar(x - width / 2, df["Original (ms)"], width, label='Original')
bar2 = ax.bar(x + width / 2, df["Optimized (ms)"], width, label='Optimized')

# 标注加速比
for i in range(len(x)):
    speedup = df["Speedup"][i]
    ax.text(x[i] + width / 2,
            df["Optimized (ms)"][i] + 1000,
            f"{speedup:.2f}×",
            ha='center',
            va='bottom',
            fontsize=8,
            rotation=45)

# 图表细节
ax.set_ylabel("Runtime (ms)")
ax.set_xlabel("Circuit")
ax.set_title("Comparison of Original and Optimized Runtime per Circuit")
ax.set_xticks(x)
ax.set_xticklabels(df["Circuit"])
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5, axis='y')

plt.tight_layout()
plt.savefig("runtime_comparison.pdf", bbox_inches="tight", dpi=300)
plt.show()
