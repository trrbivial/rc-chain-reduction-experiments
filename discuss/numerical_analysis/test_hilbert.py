import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift

# 参数设置
N = 4096  # 频率点数
domega = 0.5  # 频率分辨率
omega = np.linspace(0, (N - 1) * domega, N)  # 正频率（单边谱）
RC = 1

# 构造频域函数（仅正频率部分）
Yw_half = 1 / (1 + 1j * omega * RC)

# 构造完整频谱（解析信号：负频率部分为零）
Yw_full = np.zeros(2 * N, dtype=complex)
Yw_full[:N] = Yw_half  # 正频率
Yw_full[N:] = 0  # 负频率为零（保证因果）

# 时间轴
T = 2 * np.pi / domega  # 总时间跨度
dt = T / (2 * N)
t = np.arange(-T / 2, T / 2, dt)

# 逆变换得到时域函数
Yt = fftshift(ifft(ifftshift(Yw_full))) / dt  # 注意缩放系数

# 画图
plt.figure(figsize=(8, 4))
plt.plot(t, np.real(Yt), label="Re[Y(t)]")
#plt.plot(t, np.imag(Yt), label="Im[Y(t)]", linestyle='--')
plt.axvline(0, color='gray', linestyle=':')
plt.title("Impulse Response of 1 / (1 + jωRC)")
plt.xlabel("t")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.xlim(-5 * RC, 10 * RC)
plt.show()
