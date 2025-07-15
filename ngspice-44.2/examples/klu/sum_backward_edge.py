import os


def count_b_in_top_sort_section():
    total_b_count = 0

    # 遍历并排序所有 *_struct.out 文件
    files = sorted(f for f in os.listdir('.') if f.endswith('_struct.out'))

    for file_name in files:
        with open(file_name, 'r') as f:
            lines = f.readlines()

        start_idx = None
        end_idx = None

        # 定位 .TopSortForwardBackwardEdge 和下一个 .END
        for i, line in enumerate(lines):
            if '.TopSortForwardBackwardEdge' in line:
                start_idx = i
            elif start_idx is not None and '.END' in line:
                end_idx = i
                break

        # 提取该区域并统计 'b'
        if start_idx is not None and end_idx is not None:
            section_lines = lines[start_idx + 1:end_idx]
            section_text = ''.join(section_lines)
            b_count = section_text.count('b')
            total_b_count += b_count
            print(f"{file_name}: b count = {b_count}")
        else:
            print(f"{file_name}: Section not found")

    print(f"\nTotal 'b' count across all files: {total_b_count}")


# 调用函数
count_b_in_top_sort_section()
