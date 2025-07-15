import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def process_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # 获取矩阵大小 n（第二行第一个整数）
    n_line = lines[1].strip()
    n = int(n_line.split()[0])

    elements = set()
    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
        a, b = map(int, line.split())
        if a == 0 and b == 0:
            break
        elements.add((a, b))

    total_elements = len(elements)

    # 查找无向边
    undirected_edges = set()
    directed_edges = set()

    for a, b in elements:
        if a == b:
            continue
        if (b, a) in elements:
            undirected_edges.add(tuple(sorted((a, b))))
        else:
            directed_edges.add((a, b))

    num_undirected = len(undirected_edges)
    num_directed = len(directed_edges)

    return n, num_undirected, num_directed, total_elements, elements


def plot_sparse_matrix(elements, n, file_name, m=128):
    """
    根据稀疏矩阵元素位置绘制图像，元素位置越多的区域越深色。
    """
    # 创建一个 m*m 的空图像
    image = np.zeros((m, m), dtype=int)
    meta = file_name.split('_')[0]

    b = (n + m - 1) // m

    # 遍历所有元素位置
    for r, c in elements:
        # 将稀疏矩阵的位置映射到图像坐标
        image[r // b, c // b] = 10

    plt.close('all')
    plt.figure(figsize=(3, 3))
    # 使用 matplotlib 绘制图像
    plt.imshow(image, cmap='Reds', interpolation='nearest')  # 颜色翻转，背景白色，越多越红
    # plt.colorbar(label="Density")
    plt.title("{}".format(meta))
    print(meta + "_spm.png")
    plt.axis('off')
    plt.savefig(meta + "_spm.png", dpi=300)


# 处理所有 *_cktmatrix.txt 文件
files = sorted(f for f in os.listdir('.') if f.endswith('_cktmatrix.txt'))

headers = []
n_list = []
m_list = []
e_list = []

for file in files:
    n, num_undirected, num_directed, total_elements, elements = process_file(
        file)
    print(f"\n文件: {file}")
    print(f"  矩阵大小 n: {n}")
    print(f"  无向边数量: {num_undirected}")
    print(f"  有向边数量: {num_directed}")
    print(f"  元素总数: {total_elements}")
    #plot_sparse_matrix(elements, n, file)
    with open(file, 'r') as f:
        lines = f.readlines()

    meta = file.split('_')[0]
    headers.append(meta)
    n_list.append(str(n))
    m_list.append(str(num_undirected))
    e_list.append(str(total_elements))

latex = "\\begin{tabular}{lccccc}\n"
latex += "\\toprule\n"
latex += "电路名 & 功能 & 文件行数 & N & M & nnz \\\\\n"
latex += "\\midrule\n"
for i in range(len(headers)):
    name = headers[i]
    n = n_list[i]
    m = m_list[i]
    nnz = e_list[i]
    latex += f"{name} & & & {n} & {m} & {nnz} \\\\\n"
latex += "\\bottomrule\n"
latex += "\\end{tabular}"
print(latex)
