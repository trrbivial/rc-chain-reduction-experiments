import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt

R = 1
C = 1
n = 2


def Y(w):
    w = np.where(w == 0, 1e-8, w)
    a = 1 + 1j * R * C * w / 2 + np.sqrt(-(R * C * w)**2 / 4 + 1j * R * C * w)
    b = 1 + 1j * R * C * w / 2 - np.sqrt(-(R * C * w)**2 / 4 + 1j * R * C * w)
    tau = 1 + 1j * R * C * w
    t = b / a
    tmp = (tau * b - 1) / (tau * a - 1) * (t**(n - 1))
    ans = ((tau - b) + (a - tau) * tmp) / (1 - tmp)
    return ans / R


# 原始频域函数：例如 LPF
omega = np.linspace(-1000, 1000, 4096)
RC = 1
Yw = Y(omega)
# Yw = 1 / (1 + 1j * omega * RC)

# 使用 Hilbert 构造因果频谱
Yw_causal = Yw - np.imag(hilbert(np.real(Yw)))

# 傅立叶逆变换（注意 fftshift）
dt = 0.0001
t = np.fft.fftshift(
    np.fft.fftfreq(len(omega), d=(omega[1] - omega[0]) / (2 * np.pi)))
Yt = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Yw_causal)))

# 画图
plt.plot(t, np.real(Yt), label="Re[Y(t)] (Causal)")
plt.axvline(0, color='gray', linestyle='--')
plt.legend()
plt.xlabel("t")
plt.grid()
plt.show()
