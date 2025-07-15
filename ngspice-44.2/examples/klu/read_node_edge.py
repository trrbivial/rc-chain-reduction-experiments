import os
import re

fields = [
    ".GeneralGraphNodeEdge", ".FindCutPointNodeEdge",
    ".GlobalMinCutTotalNodeEdge", ".GlobalMinCutVddGndMaxflow",
    ".GlobalMinCutOfCenter"
]

files = sorted(f for f in os.listdir('.') if f.endswith('_struct.out'))

results = {}

for file in files:
    with open(file, 'r') as f:
        lines = f.readlines()

    file_result = {}
    for i, line in enumerate(lines):
        line = line.strip()
        if line in fields:
            if i + 1 < len(lines):
                # 读取 n, m
                match = re.match(r"(\d+)\s+(\d+)", lines[i + 1].strip())
                if match:
                    n, m = int(match.group(1)), int(match.group(2))
                    # 初始化字段数据
                    data = {"n": n, "m": m}
                    if line == ".GeneralGraphNodeEdge" and i + 2 < len(lines):
                        # 尝试读取第二行的第一个整数 c
                        second_line = lines[i + 2].strip()
                        match_c = re.match(r"(\d+)", second_line)
                        if match_c:
                            c = int(match_c.group(1))
                            data["c"] = c
                            data["c/m"] = c / m if m != 0 else None
                    file_result[line] = data
                else:
                    c = int(lines[i + 1])
                    data = {"c": c}
                    file_result[line] = data

    results[file] = file_result

headers = []
c_list = []
# 输出整理结果
for file, field_data in results.items():
    print(f"\n文件: {file}")
    for field in fields:
        if field in field_data:
            data = field_data[field]
            if 'n' in data:
                output = f"  {field}: n = {data['n']}, m = {data['m']}"
            else:
                output = f"  {field}: c = {data['c']}"
                if data['c'] > 2:
                    headers.append(file.split('_')[0])
                    c_list.append(str(data['c']))

            if field == ".GeneralGraphNodeEdge":
                if "c" in data:
                    output += f", c = {data['c']}, c/m = {data['c/m']:.4f}"
                else:
                    output += ", c = 未找到"
            print(output)
        else:
            print(f"  {field}: 未找到")

latex = "\\begin{tabular}{l" + "c" * len(headers) + "}\n"
latex += "\\toprule\n"
latex += "电路名 & " + " & ".join(headers) + " \\\\\n"
latex += "\\midrule\n"
latex += "最小割 & " + " & ".join(c_list) + " \\\\\n"
latex += "\\bottomrule\n"
latex += "\\end{tabular}"

print(latex)

import os
import re

# 查找所有 *_struct.out 文件
files = sorted(f for f in os.listdir('.') if f.endswith('_struct.out'))

# 用于存储表格数据
headers = []
n_list = []
m_list = []
c_list = []
cm_list = []

for file in files:
    with open(file, 'r') as f:
        lines = f.readlines()

    meta = file.split('_')[0]
    headers.append(meta)

    # 查找 .GeneralGraphNodeEdge
    for i, line in enumerate(lines):
        if line.strip() == '.FindCutPointNodeEdge':
            # 第1行：n, m
            if i + 1 < len(lines):
                match = re.match(r"(\d+)\s+(\d+)", lines[i + 1].strip())
                if match:
                    n, m = int(match.group(1)), int(match.group(2))
                else:
                    n, m = None, None
            else:
                n, m = None, None

            # 第2行：c
            if i + 2 < len(lines):
                match_c = re.match(r"(\d+)", lines[i + 2].strip())
                if match_c:
                    c = int(match_c.group(1))
                else:
                    c = None
            else:
                c = None

            # 填入表格数据
            n_list.append(str(n) if n is not None else "-")
            m_list.append(str(m) if m is not None else "-")
            #c_list.append(str(c) if c is not None else "-")
            #cm_val = f"{c/m:.4f}" if c is not None and m not in (None,
            #                                                     0) else "-"
            #cm_list.append(cm_val)
            break

# 生成 LaTeX 表格字符串
latex = "\\begin{tabular}{l" + "c" * len(headers) + "}\n"
latex += "\\toprule\n"
latex += "电路名 & N & M \\\\\n"
latex += "\\midrule\n"
for i in range(len(headers)):
    latex += headers[i] + " & " + n_list[i] + " & " + m_list[i] + "\\\\\n"
#latex += "c & " + " & ".join(c_list) + " \\\\\n"
#latex += "c/m & " + " & ".join(cm_list) + " \\\\\n"
latex += "\\bottomrule\n"
latex += "\\end{tabular}"

# 输出
print(latex)
