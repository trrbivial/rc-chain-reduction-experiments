import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 创建 10x20 的随机矩阵作为例子
data = np.random.rand(10, 20)

# 绘制热力图
sns.heatmap(data, cmap="coolwarm", cbar=True)

plt.title("Example Heatmap")
plt.show()
