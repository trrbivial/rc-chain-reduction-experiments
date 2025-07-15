import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

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


print(Y(-1e1), Y(1e1))
print(Y(-1e2), Y(1e2))
print(Y(-1e3), Y(1e3))
assert False

# 参数
RC = 1
t = np.linspace(-10, 10, 100)
omega = np.linspace(-50000, 50000, 4096000)
domega = omega[1] - omega[0]

pi = np.acos(-1)

# 频域传递函数
H = 1 / (1 + 1j * omega * RC)
H = -1j * np.sqrt(pi) / 2 * omega * np.exp(-omega**2 / 4)
# H = Y(omega)

N = len(H)
Yw_single_sided = np.zeros_like(H, dtype=complex)
Yw_single_sided[N // 2:] = 2 * H[N // 2:]  # 双边谱 → 单边谱
Yw_single_sided[N // 2] /= 2  # 直流点不要乘2

H = Yw_single_sided
# H = hilbert(np.real(H))

# 傅立叶逆变换：h(t) = ∫ H(jω) e^{jωt} dω / 2π
h_t = []
for ti in t:
    integrand = H * np.exp(1j * omega * ti)
    tmp = np.trapz(integrand, dx=domega) / (2 * np.pi)
    h_t.append(tmp)

h_t = np.real(h_t)

# 理论解
h_theory = np.exp(-t / RC)
h_theory = t * np.exp(-t**2)

# 画图
plt.plot(t, h_t, label="Numerical iFFT")
#plt.plot(t, h_theory, label="Analytical $e^{-t/RC}$", linestyle='--')
plt.legend()
plt.xlabel("t")
plt.ylabel("h(t)")
plt.title("Inverse Fourier Transform vs Exact")
plt.grid()
plt.show()
