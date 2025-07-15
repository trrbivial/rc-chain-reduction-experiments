import os
import matplotlib.pyplot as plt

# 获取所有 *_struct.out 文件，并按字典序排序
files = sorted(f for f in os.listdir('.') if f.endswith('_struct.out'))

meta_list = []
value_list = []
c_list = []
n_list = []

for file_name in files:
    meta = file_name.split('_')[0]
    with open(file_name, 'r') as f:
        lines = f.readlines()

    cut_start = None
    cut_end = None
    for i, line in enumerate(lines):
        if '.FindCutPointNodeEdge' in line:
            cut_start = i
        elif cut_start is not None and '.END' in line:
            cut_end = i
            break

    target_line = lines[cut_start + 1].strip().split()
    assert len(target_line) == 2
    n = int(target_line[0])
    n_list.append(n)

    # 寻找 .CutPoint 和对应的 .END
    cut_start = None
    cut_end = None
    for i, line in enumerate(lines):
        if '.CutPoint' in line:
            cut_start = i
        elif cut_start is not None and '.END' in line:
            cut_end = i
            break

    # 提取第二行的第四个字段
    if cut_start is not None and cut_end is not None and cut_end > cut_start + 1:
        target_line = lines[cut_start + 2].strip().split()
        if len(target_line) >= 4:
            try:
                chain_sum = int(target_line[1])
                value = 1.0 * chain_sum / n
                meta_list.append(meta)
                value_list.append(value)
                c_list.append(chain_sum)
            except ValueError:
                print(f"无法解析浮点数：{target_line[3]} in file {file_name}")
        else:
            print(f"第 cut 部分第二行字段不足 4 个：{file_name}")
    else:
        print(f"文件 {file_name} 未找到有效的 .CutPoint 到 .END 区段")

# 生成 LaTeX 表格字符串
latex = "\\begin{tabular}{lcccc}\n"
latex += "\\toprule\n"
latex += "电路名 & N & $N_c$ & 比例\\\\\n"
latex += "\\midrule\n"
for i in range(len(meta_list)):
    latex += meta_list[i] + " & " + str(n_list[i]) + " & " + str(
        c_list[i]) + " & " + str("{:.3f}".format(value_list[i])) + "\\\\\n"
latex += "\\bottomrule\n"
latex += "\\end{tabular}"

# 输出
print(latex)

plt.rcParams.update({
    "axes.labelsize": 14,  # 坐标轴标签字号
    "ytick.labelsize": 12,  # y轴刻度字号
    "xtick.labelsize": 12,  # y轴刻度字号
})

sum = 0.
for a in value_list:
    sum += a

sum /= len(value_list)
print(sum)
assert False

# 绘制柱状图
plt.figure(figsize=(6, 4))
plt.bar(meta_list, value_list, color='lightblue', edgecolor='black')
plt.ylabel('Split Ratio')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('split_ratio_barchart.png', dpi=300)
plt.show()
