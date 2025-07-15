import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def generate_shaded_colors():
    # 分配：6 Reds, 6 Greens, 5 Blues
    reds = [cm.Reds(x) for x in np.linspace(0.3, 0.9, 6)]  # 从较浅到较深
    greens = [cm.Greens(x) for x in np.linspace(0.3, 0.9, 6)]
    blues = [cm.Blues(x) for x in np.linspace(0.3, 0.9, 5)]

    # 合并为一个颜色列表（每个是 RGBA，可用于 matplotlib）
    colors = reds + greens + blues
    return colors


# 使用示例：绘制 17 条彩色柱子
colors = generate_shaded_colors()
x = list(range(1, 18))
y = np.random.randint(1, 10, size=17)

plt.figure(figsize=(10, 5))
plt.bar(x, y, color=colors, edgecolor='black')
plt.xticks(x)
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Custom Colored Bars")
plt.tight_layout()
plt.show()
