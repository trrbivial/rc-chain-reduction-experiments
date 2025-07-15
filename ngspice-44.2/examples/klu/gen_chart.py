import json
import sys
import os
import pandas as pd


def extract_data(json_file, output_csv="output.csv"):
    # 读取 JSON 文件
    with open(json_file, "r") as f:
        data = json.load(f)

    rows = []

    # 遍历 JSON 数据
    for path, values in data.items():
        # 提取文件名部分
        filename = os.path.basename(path)
        circuit_name = filename.split(".")[0]

        # 提取 Matrix factor time 和 Matrix solve time
        factor_time = values.get("Matrix factor time", None)
        solve_time = values.get("Matrix solve time", None)

        rows.append([circuit_name, factor_time, solve_time])

    # 创建 DataFrame
    df = pd.DataFrame(
        rows,
        columns=["Circuit Name", "Matrix Factor Time", "Matrix Solve Time"])

    # 按 Circuit Name 排序
    df = df.sort_values(by=["Circuit Name"])

    # 保存为 CSV
    df.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv}")

    return df


# 示例调用
def main(json_file):
    df = extract_data(json_file)
    print(df)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 gen_chart.py <json>")
        sys.exit(1)
    main(sys.argv[1])
