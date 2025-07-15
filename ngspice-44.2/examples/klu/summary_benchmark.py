import os


def get_line_count(file_path):
    """获取文件的行数"""
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)


def main():
    """遍历目录，查找文件并处理"""

    # 遍历当前目录及子目录
    for root, _, files in os.walk(os.getcwd()):
        for file in files:
            if file.endswith((".net", ".cir", ".sp")):
                file_path = os.path.join(root, file)

                # 针对 .net 文件
                if file.endswith(".net"):
                    file_lines = get_line_count(file_path)
                    num_files = {}
                    for file2 in files:
                        if file2.startswith("cktmatrix") and file2.endswith(
                                ".txt"):
                            try:
                                num = int(file2.split('x')[1].split('.')[0])
                                num_files[num] = os.path.join(root, file2)
                            except (ValueError, IndexError):
                                continue
                    if file.endswith("_ann.net"):
                        if num_files:
                            num = max(num_files.keys())
                            spm = num_files[num]
                            line_count = get_line_count(spm)
                            print(
                                f"{file}: filelines ~ {file_lines}, node ~ {num}, net ~ {line_count}"
                            )
                        else:
                            print(
                                f"{file}: filelines ~ {file_lines}, node ~ ?, net ~ ?"
                            )
                    else:
                        if num_files:
                            num = min(num_files.keys())
                            spm = num_files[num]
                            line_count = get_line_count(spm)
                            print(
                                f"{file}: filelines ~ {file_lines}, node ~ {num}, net ~ {line_count}"
                            )
                        else:
                            print(
                                f"{file}: filelines ~ {file_lines}, node ~ ?, net ~ ?"
                            )


if __name__ == "__main__":
    main()
