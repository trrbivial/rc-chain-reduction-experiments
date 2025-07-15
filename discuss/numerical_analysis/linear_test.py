import numpy as np
import matplotlib.pyplot as plt

# 参数设置
s = 1e-9  # 截断点
Omega_max = 1e11  # 最大频率范围
N = 5000  # 频率采样点数（越多越准）

# 频域采样点（双边谱）
omega = np.linspace(-Omega_max, Omega_max, N)
domega = omega[1] - omega[0]
jomega = 1j * omega

# 避免除以0
jomega = np.where(omega == 0, 1e-20j, jomega)

# 频谱表达式
F_omega = (1 - np.exp(-jomega * s) *
           (1 + jomega * s)) / (jomega**
                                2) + s * (1 - np.exp(-jomega * s)) / jomega

# 逆变换时域采样点
t = np.linspace(-0.5e-9, 1.5e-9, 1000)
f_t = np.zeros_like(t, dtype=np.complex128)

# 数值积分实现傅立叶逆变换
for i in range(len(t)):
    integrand = F_omega * np.exp(1j * omega * t[i])
    f_t[i] = np.trapz(integrand, omega) / (2 * np.pi)

# 绘图
plt.figure(figsize=(8, 4))
plt.plot(t * 1e9, np.real(f_t), label='Re[f(t)]')
plt.plot(t * 1e9, np.imag(f_t), '--', label='Im[f(t)]')
plt.xlabel('Time t (ns)')
plt.ylabel('f(t)')
plt.title('Inverse Fourier Transform of $F(\\omega)$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
