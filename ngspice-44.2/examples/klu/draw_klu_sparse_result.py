import json
import os
import matplotlib.pyplot as plt
import numpy as np


def load_times(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    times = {}
    for filepath, metrics in data.items():
        # 提取文件名并移除扩展名
        filename = os.path.basename(filepath)
        label = filename.split('_')[0].split('.')[0]

        factor_time = metrics.get('Matrix factor time', 0.0)
        solve_time = metrics.get('Matrix solve time', 0.0)
        total_time = factor_time + solve_time

        times[label] = total_time
    return times


# 读取 KLU 和 Sparse 1.3 的时间数据
klu_times = load_times('ngspice_times.json')
sparse_times = load_times('ngspice_times_bak.json')

# 合并所有 label，按 label 排序
all_labels = sorted(set(klu_times.keys()) | set(sparse_times.keys()))

klu_vals = [klu_times.get(label, 0.0) for label in all_labels]
sparse_vals = [sparse_times.get(label, 0.0) for label in all_labels]

# 绘图（论文格式）
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 300,
    "figure.figsize": (6, 2.5)
})

x = np.arange(len(all_labels))
width = 0.35

fig, ax = plt.subplots()
ax.bar(x - width / 2,
       sparse_vals,
       width,
       label='Sparse 1.3',
       color='#1f77b4',
       hatch='//')
ax.bar(x + width / 2,
       klu_vals,
       width,
       label='KLU',
       color='#ff7f0e',
       hatch='\\\\')

plt.yscale('log')

ax.set_xlabel('Circuit')
ax.set_ylabel('Total Time (s)')
ax.set_title('Comparison of Simulation Time')
ax.set_xticks(x)
ax.set_xticklabels(all_labels, rotation=0)
ax.legend()
ax.grid(axis='y', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig("klu_sparse_runtime_comparison.pdf")
plt.show()
