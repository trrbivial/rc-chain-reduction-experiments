import os
import re
from collections import defaultdict

# 数据结构：{meta: [(total, [(name, number), ...])]}
data = []

# 遍历目录中的文件
for file_name in os.listdir('.'):
    if file_name.endswith('_struct.out'):
        meta = file_name.split('_')[0]
        with open(file_name, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if '.DevTypeR' in line:
                # 找到.DevTypeR后第一个.END
                for j in range(i + 1, len(lines)):
                    if '.END' in lines[j]:
                        block = lines[i + 1:j]
                        break

                header = block[0].strip()
                match = re.match(r'(\d+)\s+(\d+)', header)
                n, total = int(match.group(1)), int(match.group(2))

                entries = []
                for line in block[1:1 + n]:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        name, number = parts
                        entries.append((name, number))

                data.append((meta, total, entries))
                break

print(data)


# LaTeX 表格生成
def to_latex(data):
    lines = [
        r"\begin{tabular}{|c|c|c|c|}", r"\hline",
        r"Meta & Total & Name & Number \\", r"\hline"
    ]

    for meta, total, entries in data:
        rowspan = len(entries)
        for idx, (name, number) in enumerate(entries):
            row = []
            if idx == 0:
                row.append(r"\multirow{%d}{*}{%s}" % (rowspan, meta))
                row.append(r"\multirow{%d}{*}{%s}" % (rowspan, total))
            row.append(name)
            row.append(number)
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    return "\n".join(lines)


latex_output = to_latex(data)
print(latex_output)

# 写入输出文件
with open('table_output.tex', 'w') as f:
    f.write(latex_output)

print("LaTeX 表格已生成到 table_output.tex")
