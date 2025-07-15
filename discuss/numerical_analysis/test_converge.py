import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftshift, ifftshift, ifft, fftfreq

# 设置频率轴
N = 40960  # 频点数
dx = 0.05  # 频域采样间隔
x = np.linspace(-N // 2, N // 2 - 1, N) * dx  # 对应频率（等价于 omega）

# 定义频域函数 F(x)
numerator = np.exp(-1j * x) * (1 + 1j * x) - 1
denominator = x**2
# 避免 x=0 导致除以 0，使用 l'Hôpital 法则结果处理
denominator[np.abs(x) < 1e-12] = 1  # 临时填充防止除 0
Fx = numerator / denominator
Fx[np.abs(x) < 1e-12] = 0  # x=0 时极限为 0

Fx *= np.exp(-(x / 2000)**2)  # 加上 Gauss 衰减以增强收敛性

# 执行逆傅立叶变换
ft = fftshift(ifft(ifftshift(Fx))) / dx  # 注意缩放因子
t = fftshift(fftfreq(N, d=dx)) * 2 * np.pi  # 生成时间轴

x = []
y = []
for i in range(len(t)):
    if -0.01 < t[i] < 2:
        x.append(t[i])
        y.append(np.real(ft[i]))

# 绘图
plt.figure(figsize=(8, 4))
plt.plot(x, y, label="Re[f(t)]")
#plt.plot(t, np.real(ft), label="Re[f(t)]")
#plt.plot(t, np.imag(ft), label="Im[f(t)]", linestyle="--")
plt.axvline(0, color='gray', linestyle=':')
plt.title("Inverse Fourier Transform of F(x)")
plt.xlabel("t")
plt.ylabel("f(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
