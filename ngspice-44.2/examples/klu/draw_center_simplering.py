import os
import re
import matplotlib.pyplot as plt

# 存储每个文件提取的数据：{meta: [(c, number), ...]}
data = {}

# 获取所有 *_struct.out 文件，并按字典序排序
file_list = sorted([f for f in os.listdir('.') if f.endswith('_struct.out')])

for file_name in file_list:
    meta = file_name.split('_')[0]
    with open(file_name, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if '.CenterCliqueSummarySimpleRing' in line:
            # 找到 .END
            for j in range(i + 1, len(lines)):
                if '.END' in lines[j]:
                    block = lines[i + 1:j]
                    break
            else:
                continue  # 如果没有 .END 就跳过

            if not block:
                continue

            pairs = []
            for line in block:
                parts = line.strip().split()
                assert len(parts) == 2
                d, sz = int(parts[0]), int(parts[1])
                pairs.append((d, sz))

            data[meta] = sorted(pairs)
            break  # 每个文件只处理一个 .DevTypeC 块

colors = plt.cm.get_cmap("tab20")

idx = 0
for file_name in file_list:
    meta = file_name.split('_')[0]
    plt.close('all')  # 安全起见，关闭所有旧图
    plt.figure(figsize=(3, 3))

    for name, pairs in data.items():
        if meta == name:
            x = range(3, pairs[len(pairs) - 1][0] + 1)
            y = [0 for i in range(len(x))]
            cs = [p[0] for p in pairs]
            nums = [p[1] for p in pairs]
            for p in pairs:
                y[p[0] - 3] += p[1]
            plt.plot(cs, nums, marker='*', color=colors(idx))

    idx += 1

    plt.yscale('log')
    plt.xlabel('Size of Simple Ring')
    plt.ylabel('Count')
    plt.title(meta)
    plt.grid(True)
    plt.tight_layout()

    current_ticks = plt.gca().get_xticks()
    s = set(current_ticks.tolist() + [3])
    if -20 in s:
        s.remove(-20)
    new_ticks = sorted(s)

    plt.xticks(new_ticks)

    print(meta + "_centersimplering.png")
    plt.savefig(meta + "_centersimplering.png", dpi=300)
