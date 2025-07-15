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
        if '.DevTypeC' in line:
            # 找到 .END
            for j in range(i + 1, len(lines)):
                if '.END' in lines[j]:
                    block = lines[i + 1:j]
                    break
            else:
                continue  # 如果没有 .END 就跳过

            if not block:
                continue
            # 第一行为 n 和 total
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
                    # 去掉末尾的 f 并转换为 float
                    c = float(c_str.rstrip('f'))
                    number = int(number_str)
                    if 0 <= c <= 2:
                        pairs.append((c, number))
                except ValueError:
                    continue

            data[meta] = sorted(pairs)
            break  # 每个文件只处理一个 .DevTypeC 块

colors = plt.cm.get_cmap("tab20")

mi = 2
mx = 0
idx = 0
for file_name in file_list:
    meta = file_name.split('_')[0]
    plt.close('all')  # 安全起见，关闭所有旧图
    plt.figure(figsize=(3, 3))

    plt.xlim(0, 2 * 1e-15)
    plt.ylim(1, 40000)
    for name, pairs in data.items():
        if meta == name:
            cs = [p[0] * 1e-15 for p in pairs]
            nums = [p[1] for p in pairs]
            plt.plot(cs, nums, marker='o', color=colors(idx))

    idx += 1
    for c in cs:
        mi = min(mi, c)
        mx = max(mx, c)

    sum = 0
    for num in nums:
        sum += num

    print(sum)

    plt.yscale('log')
    plt.xlabel('Farad (F)')
    plt.ylabel('Quantity')
    plt.title(meta)
    plt.grid(True)
    plt.tight_layout()
    print(meta + "_devC_1e-15.png")
    plt.savefig(meta + "_devC_1e-15.pdf")

print(mi, mx)
