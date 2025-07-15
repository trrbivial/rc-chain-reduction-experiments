import os
import re
import matplotlib.pyplot as plt

# 保存所有不同的字符串 s
s_set = set()
np_set = set()

# 获取所有 *_struct.out 文件并排序
file_list = sorted([f for f in os.listdir('.') if f.endswith('_struct.out')])

for file_name in file_list:
    with open(file_name, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if '.DevTypeM' in line:
            # 找到 .END 的行
            for j in range(i + 1, len(lines)):
                if '.END' in lines[j]:
                    block = lines[i + 1:j]
                    break
            else:
                continue  # 如果找不到 .END，跳过这个块

            if not block:
                continue

            # 第一行为 n 和 total，跳过
            header = block[0].strip()
            match = re.match(r'(\d+)\s+(\d+)', header)
            if not match:
                continue
            n = int(match.group(1))

            # 处理后面 n 行
            for line in block[1:1 + n]:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue  # 至少要两个字段，才能去掉一个
                s = ' '.join(parts[:-1])
                np = ' '.join(parts[1:-1])
                s_set.add(s)
                np_set.add(np)

            break  # 每个文件只处理一个 .DevTypeM 区块

# 输出所有本质不同的字符串
print("本质不同的 s 共 {} 个：".format(len(s_set)))
for s in sorted(s_set):
    print(s)
print("忽略 n/p 本质不同的 s 共 {} 个：".format(len(np_set)))
for np in sorted(np_set):
    print(np)

mp = {
    "l=0.12u w=0.33u as=0.08745p ad=0.08745p ps=1.19u pd=1.19u": 1,
    "l=0.12u w=0.44u as=0.1166p ad=0.1166p ps=1.41u pd=1.41u": 2,
    "l=0.12u w=0.495u as=0.131175p ad=0.131175p ps=1.52u pd=1.52u": 3,
    "l=0.12u w=0.55u as=0.14575p ad=0.14575p ps=1.63u pd=1.63u": 4,
    "l=0.12u w=0.605u as=0.160325p ad=0.160325p ps=1.74u pd=1.74u": 5,
    "l=0.12u w=0.66u as=0.1749p ad=0.1749p ps=1.85u pd=1.85u": 6,
    "l=0.12u w=0.715u as=0.189475p ad=0.189475p ps=1.96u pd=1.96u": 17,
    "l=0.12u w=0.77u as=0.20405p ad=0.20405p ps=2.07u pd=2.07u": 8,
    "l=0.12u w=0.88u as=0.2332p ad=0.2332p ps=2.29u pd=2.29u": 7,
    "l=0.12u w=0.935u as=0.247775p ad=0.247775p ps=2.4u pd=2.4u": 9,
    "l=0.12u w=0.99u as=0.26235p ad=0.26235p ps=2.51u pd=2.51u": 10,
    "l=0.12u w=1.045u as=0.276925p ad=0.276925p ps=2.62u pd=2.62u": 11,
    "l=0.12u w=1.155u as=0.306075p ad=0.306075p ps=2.84u pd=2.84u": 12,
    "l=0.12u w=1.1u as=0.2915p ad=0.2915p ps=2.73u pd=2.73u": 16,
    "l=0.12u w=1.375u as=0.364375p ad=0.364375p ps=3.28u pd=3.28u": 13,
    "l=0.12u w=1.485u as=0.393525p ad=0.393525p ps=3.5u pd=3.5u": 14,
    "l=0.12u w=1.54u as=0.4081p ad=0.4081p ps=3.61u pd=3.61u": 15,
}

# 存储每个文件提取的数据：{meta: [(c, number), ...]}
data = {}

# 获取所有 *_struct.out 文件，并按字典序排序
file_list = sorted([f for f in os.listdir('.') if f.endswith('_struct.out')])

for file_name in file_list:
    meta = file_name.split('_')[0]
    with open(file_name, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if '.DevTypeM' in line:
            # 找到 .END
            for j in range(i + 1, len(lines)):
                if '.END' in lines[j]:
                    block = lines[i + 1:j]
                    break
            else:
                continue  # 如果没有 .END 就跳过

            if not block:
                continue
            # 第一行为 n 和 total，跳过
            header = block[0].strip()
            match = re.match(r'(\d+)\s+(\d+)', header)
            if not match:
                continue
            n = int(match.group(1))

            pairs = []
            # 处理后面 n 行
            for line in block[1:1 + n]:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue  # 至少要两个字段，才能去掉一个
                s = ' '.join(parts[1:-1])
                idx = mp[s]
                assert 1 <= idx <= 17
                number = int(parts[len(parts) - 1])
                pairs.append((idx, number))

            data[meta] = sorted(pairs)

            break  # 每个文件只处理一个 .DevTypeM 区块

colors = plt.cm.get_cmap("tab20")

idx = 0
for file_name in file_list:
    meta = file_name.split('_')[0]
    plt.close('all')  # 安全起见，关闭所有旧图
    plt.figure(figsize=(3, 3))

    x = list(range(1, 18))
    y = [0 for i in range(1, 18)]
    plt.xlim(0, 18)
    #plt.ylim(1, 40000)
    for name, pairs in data.items():
        if meta == name:
            for p in pairs:
                #print(p[0], p[1])
                y[p[0] - 1] += p[1]
            plt.bar(x, y, color=colors(idx), edgecolor='black')

    idx += 1

    sum = 0
    for num in y:
        sum += num

    print(sum)

    continue

    plt.xlabel('Index')
    plt.ylabel('Size')
    plt.title(meta)
    plt.tight_layout()
    print(meta + "_devM.png")
    plt.savefig(meta + "_devM.png", dpi=300)
