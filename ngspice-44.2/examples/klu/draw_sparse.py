import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def read_sparse_matrix(file_path):
    """
    读取稀疏矩阵文件，解析并返回非零元素的位置列表。
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 忽略第一行
    n = int(lines[1].strip().split()[0])  # 第二行的第一个整数表示矩阵的大小 n
    elements = []

    # 从第三行开始，读取每行的两个整数表示元素的行列位置
    for line in lines[2:]:
        r, c = map(int, line.strip().split())
        if r == 0 and c == 0:
            break
        elements.append((r - 1, c - 1))  # 转换为0索引
    return n, elements


def compute_max_distance(elements):
    """
    计算每个元素到对角线的最大距离，返回一个包含所有距离的列表。
    """
    distances = [abs(r - c) for r, c in elements]
    max_distance = max(distances) if distances else 0
    return distances, max_distance


def plot_sparse_matrix(elements, n, directory_name, m=128):
    """
    根据稀疏矩阵元素位置绘制图像，元素位置越多的区域越深色。
    """
    # 创建一个 m*m 的空图像
    image = np.zeros((m, m), dtype=int)

    b = (n + m - 1) // m

    # 遍历所有元素位置
    for r, c in elements:
        # 将稀疏矩阵的位置映射到图像坐标
        image[r // b, c // b] = 10

    # 使用 matplotlib 绘制图像
    plt.imshow(image, cmap='Reds', interpolation='nearest')  # 颜色翻转，背景白色，越多越红
    # plt.colorbar(label="Density")
    plt.title("Sparse Matrix {} Visualization".format(directory_name))
    plt.show()


# 使用示例：
def main(file_path):
    directory_name = os.path.basename(os.path.dirname(file_path))

    n, elements = read_sparse_matrix(file_path)
    distances, max_distance = compute_max_distance(elements)
    print(f"每个元素到对角线的最大距离: {max_distance}")
    plot_sparse_matrix(elements, n, directory_name)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 draw_sparse.py <matrix.txt>")
        sys.exit(1)
    main(sys.argv[1])
