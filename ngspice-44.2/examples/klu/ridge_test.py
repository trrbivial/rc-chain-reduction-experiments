import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 生成示例数据
np.random.seed(0)
data = pd.DataFrame({
    'value':
    np.concatenate(
        [np.random.normal(loc=i, scale=0.5, size=200) for i in range(5)]),
    'group':
    sum([[f'Group {i}'] * 200 for i in range(5)], [])
})

# 画出 KDE 曲线堆叠（模拟 ridge plot）
plt.figure(figsize=(10, 6))
for i, group in enumerate(sorted(data['group'].unique())):
    subset = data[data['group'] == group]
    sns.kdeplot(subset['value'],
                fill=True,
                label=group,
                bw_adjust=1,
                clip=(-2, 8),
                linewidth=1.5)
    plt.text(8.2, i * 0.12, group, va='center')  # 添加标签

plt.title("Simulated Ridge Plot")
plt.show()
