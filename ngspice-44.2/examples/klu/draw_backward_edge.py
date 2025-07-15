import os
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.cm as cm
import re
import numpy as np

# 遍历文件
files = sorted(f for f in os.listdir('.') if f.endswith('_struct.out'))


def parse_struct_block_n(lines):
    start, end = None, None
    for i, line in enumerate(lines):
        if '.TopSortStruct' in line:
            start = i + 1
        elif start != None and '.END' in line:
            end = i
            break
    if start is None or end is None:
        raise ValueError(
            "Input file must contain '.TopSortStruct' and '.END' markers.")
    return int(lines[start].strip())


def parse_struct_block(lines):
    start, end = None, None
    for i, line in enumerate(lines):
        if '.TopSortStruct' in line:
            start = i + 1
        elif start != None and '.END' in line:
            end = i
            break
    if start is None or end is None:
        raise ValueError(
            "Input file must contain '.TopSortStruct' and '.END' markers.")
    return lines[start:end]


def process_data(lines):
    n = int(lines[0].strip())
    data = [{} for _ in range(n)]  # Each element is a dict {name: size}

    one_bottom = []
    name_set = set()

    for i in range(n):
        tokens = lines[i + 1].strip().split()
        index = int(tokens[0])
        m = int(tokens[1])
        one_bottom.append(None)
        for j in range(m):
            name = tokens[2 + 2 * j]
            size = int(tokens[3 + 2 * j])
            if m == 1 and size == 1:
                one_bottom[i] = name

            data[i][name] = data[i].get(name, 0) + size
            if name not in name_set == 0:
                assert False
            name_set.add(name)

    return data, sorted(name_set), one_bottom


def draw_stacked_bar(data, name_list, one_bottom, file_name, lines):
    plt.rcParams.update({
        "axes.labelsize": 14,  # 坐标轴标签字号
        "xtick.labelsize": 12,  # x轴刻度字号
        "ytick.labelsize": 12,  # y轴刻度字号
    })
    file_name = file_name.split("_")[0]
    maxh = 320
    if file_name == "c432" or file_name == "c499":
        maxh = 480
    n = len(data)
    x = list(range(1, n + 1))
    bottoms = [0] * n

    name_list = [
        "g", "x", "y", "z", "w", "and", "nand", "or", "nor", "not", "xor",
        "netg", "netx", "nety", "netz", "netw", "gnd"
    ]
    blues = [cm.Blues(x) for x in np.linspace(0.9, 0.3, 5)]
    reds = [cm.Reds(x) for x in np.linspace(0.9, 0.3, 6)]  # 从较浅到较深
    greens = [cm.Greens(x) for x in np.linspace(0.9, 0.3, 5)]
    blacks = ['black']

    # 合并为一个颜色列表（每个是 RGBA，可用于 matplotlib）
    # colors = blues + reds + greens + blacks
    colors = ['lightgray' for _ in name_list]

    plt.close('all')
    plt.figure(figsize=(2048 / 100, maxh / 100), dpi=100)

    idx = 0
    for name in name_list:
        heights = [d.get(name, 0) for d in data]
        for i in range(n):
            if one_bottom[i] == name:
                heights[i] += 0.414
        is_all_zero = True
        for i in range(n):
            if heights[i] > 0:
                is_all_zero = False
        if not is_all_zero:
            plt.bar(x, heights, bottom=bottoms, label=name, color=colors[idx])
            bottoms = [bottoms[i] + heights[i] for i in range(n)]
        idx += 1

    plt.yscale("log", base=2)

    max_val = max(sum(d.values()) for d in data)
    plt.ylim(1, max_val)

    plt.xlabel("Layer")
    plt.ylabel("Size")
    plt.title(file_name)
    plt.legend(loc='upper right')
    plt.tight_layout()
    n = parse_struct_block_n(lines)

    edges = []
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if '.TopSortForwardBackwardEdge' in line:
            start_idx = i
        elif start_idx is not None and '.END' in line:
            end_idx = i
            break

    # 遍历区域，提取所有 b 后的 (a, b, c)
    tokens = ' '.join(lines[start_idx + 1:end_idx]).split()
    i = 0
    while i < len(tokens):
        if tokens[i] == 'b':
            try:
                a = int(tokens[i + 1])
                b = int(tokens[i + 2])
                c = int(tokens[i + 3])
                edges.append((a, b, c))
                i += 4
            except (IndexError, ValueError):
                print(f"格式错误：{file_name} 中的某个 b 后面不是三个整数")
                i += 1
        else:
            i += 1

    # 可视化绘图
    st = math.log10(max_val) / (len(edges) + 1.0)
    y_positions = [10**(st * (i + 1)) for i in range(len(edges))]
    y_positions.reverse()

    for i, (a, b, c) in enumerate(edges):
        y = y_positions[i]
        plt.annotate(
            "",
            xy=(b, y),
            xytext=(a, y),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
        )
        plt.text(a + 0.2, y, str(c), ha='center', va='bottom', fontsize=12)


# 设置图形属性
#plt.show()
    plt.savefig(file_name + "_backward.png")
    #plt.show()


def main(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    struct_lines = parse_struct_block(lines)
    data, name_list, one_bottom = process_data(struct_lines)
    draw_stacked_bar(data, name_list, one_bottom, filename, lines)


for file_name in files:
    main(file_name)
