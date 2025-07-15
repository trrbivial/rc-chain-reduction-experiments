import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 数据输入
data = {
    "Circuit": [
        'c1355', 'c1908', 'c2670', 'c3540', 'c432', 'c499', 'c5315', 'c6288',
        'c7552', 'c880'
    ],
    "Original (ms)":
    [13074, 18569, 28971, 43406, 6066, 12112, 66139, 58812, 58891, 9928],
    "Optimized (ms)":
    [11418, 16639, 23723, 41746, 5214, 11572, 59420, 56092, 58576, 8678],
}

# 转换为 DataFrame
df = pd.DataFrame(data)
df["Speedup"] = df["Original (ms)"] / df["Optimized (ms)"]

mul = 1.
cnt = 0
for x in df["Speedup"]:
    cnt += 1
    mul *= x

mul = mul**(1. / cnt)
print(mul)
