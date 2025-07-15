import matplotlib.pyplot as plt
import numpy as np

# 示例数据
x = list(range(1, 18))
y = np.random.randint(2, 10, size=17)

# 生成颜色（可按你已有的配色方案设置）
colors = plt.cm.viridis(np.linspace(0.3, 0.9, 17))

# 绘制柱状图
plt.figure(figsize=(12, 6))
bars = plt.bar(x, y, color=colors, edgecolor='black')

# 示例：从柱 b=5 向柱 a=2 连一条有向边
a_index = 2 - 1  # 因为 x 从 1 开始，而列表是从 0 开始
b_index = 5 - 1

# 获取柱子的中心坐标和高度
a_x = x[a_index]
a_y = y[a_index]

b_x = x[b_index]
b_y = y[b_index]

# 添加箭头（从 b 指向 a）
plt.annotate(
    '',  # 没有文字
    xy=(a_x, a_y + 0.5),  # 箭头尖端
    xytext=(b_x, b_y + 0.5),  # 起始点
    arrowprops=dict(arrowstyle="->", color='red', lw=2),
)

edges = [(5, 2), (7, 1), (10, 3)]
for b, a in edges:
    b_idx = b - 1
    a_idx = a - 1
    plt.annotate('',
                 xy=(x[a_idx], y[a_idx] + 0.5),
                 xytext=(x[b_idx], y[b_idx] + 0.5),
                 arrowprops=dict(arrowstyle="->", color='blue', lw=1.5))

# 基本设置
plt.xticks(x)
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Bar Chart with Directed Edge")
plt.tight_layout()
plt.ylim(0, max(y) + 3)
plt.show()
