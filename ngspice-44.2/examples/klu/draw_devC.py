import os
import re
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取所有 *_struct.out 文件，按字典序排序
file_list = sorted([f for f in os.listdir('.') if f.endswith('_struct.out')])
raw_data = {}  # {meta: [(c, number), ...]}
c_set = set()

# 2. 解析每个文件
for file_name in file_list:
    meta = file_name.split('_')[0]
    with open(file_name, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if '.DevTypeC' in line:
            for j in range(i + 1, len(lines)):
                if '.END' in lines[j]:
                    block = lines[i + 1:j]
                    break
            else:
                continue

            if not block:
                continue

            header = block[0].strip()
            match = re.match(r'(\d+)\s+(\d+)', header)
            if not match:
                continue
            n = int(match.group(1))

            pairs = []
            for line in block[1:1 + n]:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                c_str, number_str = parts
                try:
                    c = float(c_str.rstrip('f'))
                    number = int(number_str)
                    if 0 <= c <= 2:
                        pairs.append((c, number))
                        c_set.add(c)
                except ValueError:
                    continue

            raw_data[meta] = pairs
            break

# 3. 统一横轴
c_list = sorted(c_set)
c_index = {c: i for i, c in enumerate(c_list)}
x = np.arange(len(c_list))  # 均匀横坐标

# 4. 构造每个文件的 y 值（和 x 对应）
file_names = sorted(raw_data.keys())
ys = []

for meta in file_names:
    y = [0] * len(c_list)
    for c, number in raw_data[meta]:
        idx = c_index[c]
        y[idx] = number
    ys.append(y)

plt.rcParams.update({
    "axes.labelsize": 14,  # 坐标轴标签字号
    "ytick.labelsize": 12,  # y轴刻度字号
})
# 5. 绘图：层叠柱状图
plt.figure(figsize=(16, 9))
bottom = np.zeros(len(c_list))
colors = plt.cm.get_cmap("tab20")

for idx, (meta, y) in enumerate(zip(file_names, ys)):
    plt.bar(x, y, bottom=bottom, label=meta, width=0.6, color=colors(idx))
    bottom += np.array(y)

plt.yscale('log')
# 6. 美化图形
plt.xticks(x, [f"{c:.3f}" for c in c_list], rotation=45)
plt.xlabel('Farad (f)')
plt.ylabel('Size')
plt.title('Capacitance Farad-Size Ditribution of All Circuits (Stacked)')
plt.legend(title='File Meta')
plt.tight_layout()
plt.savefig('stacked_barchart.png', dpi=300)
plt.show()
