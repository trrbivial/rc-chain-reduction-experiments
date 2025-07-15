import subprocess
import numpy as np
import re


def extract_analysis_time(logfile="ngspice.log"):
    with open(logfile, "r") as f:
        content = f.read()
    # 匹配 'Total analysis time (seconds) = 0.12345'
    match = re.search(r"Total analysis time \(seconds\)\s*=\s*([0-9.]+)",
                      content)
    if match:
        return float(match.group(1))
    else:
        raise ValueError("未能在 ngspice.log 中找到仿真时间")


def run_ngspice(filename):
    # 预热一次
    subprocess.run(["ngspice", "-b", "-o", "ngspice.log", filename],
                   check=True,
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

    times = []
    for _ in range(10):
        subprocess.run(["ngspice", "-b", "-o", "ngspice.log", filename],
                       check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        t = extract_analysis_time("ngspice.log")
        times.append(t)

    avg_time = sum(times) / len(times)

    # 读取仿真输出数据（最后一次）
    data = np.loadtxt("output.dat")
    return data, avg_time


def compute_errors(std_data, ours_data):
    # 比较时间轴是否一致
    if not np.allclose(std_data[:, 0], ours_data[:, 0], rtol=1e-9, atol=1e-12):
        raise ValueError("时间轴不一致，无法比较")

    std_vals = std_data[:, 1:]
    ours_vals = ours_data[:, 1:]

    abs_error = np.abs(std_vals - ours_vals)
    rel_error = np.abs(std_vals - ours_vals) / (np.abs(std_vals) + 1e-12)

    mae = np.mean(abs_error)
    mre = np.mean(rel_error)

    return mae, mre


def main():
    with open("file_list.txt", "r") as f:
        lines = f.readlines()

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue  # 跳过空行或注释

        try:
            file_A, file_B = line.split()
            print(f"\n[{line_num}] Comparing: {file_A} vs {file_B}")

            std_data, std_time = run_ngspice(file_A)
            ours_data, ours_time = run_ngspice(file_B)

            mae, mre = compute_errors(std_data, ours_data)

            print(f"  平均绝对误差 (MAE): {mae:.4e}")
            print(f"  平均相对误差 (MRE): {mre:.4e}")
            print(f"  A 文件平均仿真时间: {std_time:.4f} 秒")
            print(f"  B 文件平均仿真时间: {ours_time:.4f} 秒")

        except Exception as e:
            print(f"  ❌ 出错: {e}")


if __name__ == "__main__":
    main()
