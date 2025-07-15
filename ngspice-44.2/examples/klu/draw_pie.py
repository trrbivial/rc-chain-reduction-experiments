import matplotlib.pyplot as plt

data = {
    "SearchForPivot ": 108.680628,
    "SearchForSingleton ": 33.793849,
    "QuicklySearchDiagonal ": 31.919616,
    "SearchDiagonal ": 42.711398,
    "FindBiggestInColExclude ": 105.554819,
    "ExchangeRowsAndCols ": 213.565274,
    "ExchangeColElements ": 101.794769,
    "ExchangeRowElements ": 111.014189,
    "RealRowColElimination ": 82.962531,
    "CreateFillin ": 3.128893,
    "spFactor ": 3538.741553,
    "spOrderAndFactor ": 405.892744,
}

# 过滤掉值为 0 的项
filtered_data = {k: v for k, v in data.items() if v > 0}
labels = list(filtered_data.keys())
values = list(filtered_data.values())

# 画饼状图
plt.figure(figsize=(10, 10))
plt.pie(values,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        wedgeprops={'edgecolor': 'black'})
plt.title("c7552 Execution Time Breakdown")
plt.show()
