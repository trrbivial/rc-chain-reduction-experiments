import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 读取 CSV 文件
data = pd.read_csv("output_temp2.csv")

# 提取数据
circuit_names = data["Circuit Name"].tolist()
matrix_factor_time = data["Matrix Factor Time(s)"].tolist()
matrix_solve_time = data["Matrix Solve Time(s)"].tolist()

# 计算总时间
total_time = np.array(matrix_factor_time) + np.array(matrix_solve_time)

# 绘制柱状图
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.7
bar1 = ax.bar(circuit_names,
              matrix_factor_time,
              bar_width,
              label="Matrix Factor Time",
              color='skyblue')
bar2 = ax.bar(circuit_names,
              matrix_solve_time,
              bar_width,
              bottom=matrix_factor_time,
              label="Matrix Solve Time",
              color='salmon')

# 旋转 x 轴标签以便阅读
plt.xticks(rotation=45, ha='right')
ax.set_ylabel("Total Time (s)")
ax.set_xlabel("Circuit Name")
ax.set_title("Matrix Factor & Solve Time for Circuits")
ax.legend()

# 显示图表
plt.show()
