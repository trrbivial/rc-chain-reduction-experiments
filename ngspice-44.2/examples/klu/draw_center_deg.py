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
        if '.CenterCliqueSummaryDeg' in line:
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
                if 2 <= d <= 100:
                    if sz == 1:
                        pairs.append((d, 1.414))
                    else:
                        pairs.append((d, sz))

            data[meta] = sorted(pairs)
            break  # 每个文件只处理一个 .DevTypeC 块

colors = plt.cm.get_cmap("tab20")

idx = 0
for file_name in file_list:
    meta = file_name.split('_')[0]
    plt.close('all')  # 安全起见，关闭所有旧图
    plt.figure(figsize=(3, 3))

    plt.xlim(0, 20)
    if meta == "c7552":
        plt.xlim(0, 80)
    plt.ylim(1, 90000)
    for name, pairs in data.items():
        if meta == name:
            cs = [p[0] for p in pairs]
            nums = [p[1] for p in pairs]
            plt.bar(cs, nums, color=colors(idx))
            plt.plot(cs, nums, alpha=0.3, color=colors(idx))
            total = sum(nums)
            target_index = 1
            for i in range(len(cs)):
                if cs[i] == 3:
                    target_index = i
                    break
            percent = nums[target_index] / total * 100
            plt.text(3,
                     nums[target_index] + 1,
                     f"{percent:.0f}%",
                     ha='center',
                     color=colors(idx))

    idx += 1

    plt.yscale('log')
    plt.xlabel('Degree')
    plt.ylabel('Size')
    plt.title(meta)
    plt.grid(True)
    plt.tight_layout()
    print(meta + "_centerdeg.png")
    plt.savefig(meta + "_centerdeg.png", dpi=300)
