import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy.integrate import quad  # 数值积分

# ------------------ 参数 ------------------
n = 2  # 节点数（V0 到 V100）
R = 1  # 电阻 Ω
C = 1e-15  # 电容 F
pi = np.acos(-1)
dt = 1e-12  # 时间步 s
M = 1e4
t_max = 1e-9  # 最大时间 s
f = 1000 * (10**6)
steps = int(t_max / dt)
time = np.linspace(0, t_max, steps)


def solve_normal(n):
    # ------------------ 初始化矩阵 ------------------
    N = n + 1  # 节点总数，包括 V0
    G = sp.lil_matrix((N, N))  # 电导矩阵
    Cmat = sp.lil_matrix((N, N))  # 电容矩阵

    # 电导矩阵 G（由电阻构成）
    for i in range(N):
        if i > 0:
            G[i, i] += 1 / R
            G[i, i - 1] -= 1 / R
            G[i - 1, i] -= 1 / R
            G[i - 1, i - 1] += 1 / R

# 电容矩阵 C（对地）
    for i in range(N):
        Cmat[i, i] = C

# 转为 CSR 格式提高求解效率
    G = G.tocsr()
    Cmat = Cmat.tocsr()

    # ------------------ 时间步进求解 ------------------
    V = np.zeros((N, steps))  # 所有节点电压随时间的矩阵

    # 初始值（t=0）
    V[:, 0] = 0

    # 向后欧拉： (C/dt + G) * V[t] = C/dt * V[t-1] + B
    A = (Cmat / dt + G)
    A = A.tocsc()

    solver = spla.factorized(A)  # LU 分解，提高效率

    iter_times = 10

    I_prime = np.zeros(steps)
    I = np.zeros(steps)

    for k in range(1, steps):
        V[:, k] = V[:, k - 1]
        t = time[k]

        v_source = np.sin(t * 2 * pi * 1e9)
        I_prime[k] = 2 * pi * 1e9 * np.cos(t * 2 * pi * 1e9)
        V[0, k] = v_source

        dV = np.zeros(N)
        print("step = ", k)
        for i in range(iter_times):
            b = -(G @ (V[:, k]) + Cmat @ (dV / dt))
            b[0] = -I_prime[k]
            solver_A = spla.factorized(A)
            dV = dV + solver_A(b)
            print(A)
            print(b)
            print(dV)
            V[:, k] = V[:, k - 1] + dV

        if k > 5:
            assert False

        for i in range(N):
            I[k] += dV[i] / dt
        print(V[:, k], I[k], I_prime[k], dV[0] / dt, dV[1] / dt, dV[2] / dt)


# ------------------ 绘图 ------------------
    plt.figure(figsize=(10, 6))
    for idx in [0, 1, 2]:
        plt.plot(time * 1e9, V[idx], label=f'V{idx}')
    #plt.plot(time * 1e9, I, label=f'I')
    plt.xlabel('Time (ns)')
    plt.ylabel('Voltage (V)')
    plt.title('Transient Response of RC Chain Driven by sin(t)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def solve_second_order(n):
    N = n + 1
    V_std = np.zeros(N, dtype="complex")
    V_verify = np.zeros(N, dtype="complex")
    V = np.zeros(N, dtype="complex")
    V[0] = V_std[0] = 1.14 + 0.514j
    I = 1.919 + 0.81j
    w = 1.2
    a = 1 - 1.0j * pi * R * C * w + np.sqrt(-(pi * R * C * w)**2 -
                                            2.0j * pi * R * C * w)
    b = 1 - 1.0j * pi * R * C * w - np.sqrt(-(pi * R * C * w)**2 -
                                            2.0j * pi * R * C * w)

    tau = 1 - 2j * pi * R * C * w
    V_std[1] = (1 - 2j * pi * R * C * w) * V_std[0] + R * I
    for i in range(2, n):
        V_std[i] = (1 + tau) * V_std[i - 1] - V_std[i - 2]

    C0 = ((b - tau) * V[0] - R * I) / (b - a)
    C1 = ((tau - a) * V[0] + R * I) / (b - a)

    print((C0 + C1 - V[0]))
    for i in range(1, n):
        V[i] = C0 * (a**i) + C1 * (b**i)

    for i in range(1, n):
        print(V[i] - V_std[i])

    fz = ((b - tau) * (tau * (a**n) - (a**(n - 1))) + (tau - a) *
          (tau * (b**n) - (b**(n - 1))))
    fm = (tau * (a**n) - (a**(n - 1))) - (tau * (b**n) - (b**(n - 1)))
    I = (V[0] / R) * fz / fm
    V_verify[0] = V[0]
    V_verify[1] = (1 - 2j * pi * R * C * w) * V[0] + R * I
    for i in range(2, N):
        V_verify[i] = (1 + tau) * V_verify[i - 1] - V_verify[i - 2]

    print(fz, fm)
    print(tau * V_verify[n] - V_verify[n - 1])


def solve_fg():

    def Y(w):
        if w == 0:
            w = 1e-8
        a = 1 - 1.0j * pi * R * C * w + np.sqrt(-(pi * R * C * w)**2 -
                                                2.0j * pi * R * C * w)
        b = 1 - 1.0j * pi * R * C * w - np.sqrt(-(pi * R * C * w)**2 -
                                                2.0j * pi * R * C * w)

        tau = 1 - 2j * pi * R * C * w
        if np.abs(a) < np.abs(b):
            t = a
            a = b
            b = t
        t = b / a
        tmp = (tau * b - 1) / (tau * a - 1) * (t**(n - 1))
        ans = ((b - tau) + (tau - a) * tmp) / (1 - tmp)
        return ans / R

    def F(lambd):
        lambd = np.where(lambd == 0, 1e-20, lambd)
        ans = ((1 + 2j * pi * lambd) * np.exp(-2j * pi * lambd) -
               1) / (4 * pi * pi * lambd * lambd)
        ans = ans * Y(lambd / dt) * np.exp(2j * pi * lambd)
        return np.imag(ans)

    def G(lambd):
        lambd = np.where(lambd == 0, 1e-20, lambd)
        ans = (1 - np.exp(-2j * pi * lambd)) / (2j * pi * lambd)

        ans = ans * Y(lambd / dt) * np.exp(2j * pi * lambd)

        return np.imag(ans)

    def adaptive_simpson(f, a, b, eps, max_recursion=20):

        def simpson(f, a, b):
            c = (a + b) / 2
            return (b - a) / 6 * (f(a) + 4 * f(c) + f(b))

        def recurse(f, a, b, eps, S, depth):
            c = (a + b) / 2
            S_left = simpson(f, a, c)
            S_right = simpson(f, c, b)
            if depth <= 0 or abs(S_left + S_right - S) < 15 * eps:
                return S_left + S_right
            return recurse(f, a, c, eps / 2, S_left, depth - 1) + recurse(
                f, c, b, eps / 2, S_right, depth - 1)

        S = simpson(f, a, b)
        return recurse(f, a, b, eps, S, max_recursion)

    Fn, Ferr = quad(F, -1e-6, 1e-6)
    Gn, Gerr = quad(G, -1e-6, 1e-6)
    print(Fn, Gn)

    for i in range(-6, 24, 1):
        Fn = quad(F, -(10**(i + 1)), -(10**i))[0] + quad(
            F, 10**i, 10**(i + 1))[0]
        Gn = quad(G, -(10**(i + 1)), -(10**i))[0] + quad(
            G, 10**i, 10**(i + 1))[0]
        print(i, Fn, Gn)

    print(Y(1e18))
    x = np.linspace(1e1, 1e6, 500000)
    y = np.zeros_like(x)
    for i in range(len(x)):
        y[i] = F(x[i])

    # ------------------ 绘图 ------------------
    plt.figure(figsize=(8, 5))
    plt.plot(x, y)
    plt.xlabel("Time (ps)")
    plt.ylabel("Fn(lambd)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def Ratio(w):
    if w == 0:
        w = 1e-8
    a = 1 - 1j * R * C * w / 2 + np.sqrt(-(R * C * w)**2 / 4 - 1j * R * C * w)
    b = 1 - 1j * R * C * w / 2 - np.sqrt(-(R * C * w)**2 / 4 - 1j * R * C * w)

    tau = 1 - 1j * R * C * w
    t = b / a
    apb = 2 - 1j * R * C * w
    tmp = (tau * b - 1) / (tau * a - 1) * (t**(n - 1))
    ans = ((b - tau) + (tau - a) * tmp) / (1 - tmp)
    print("alter = ", (apb - 2 * tau) - ((b - tau) + (tau - a) * tmp))
    print("(b-tau + (tau-a)*tmp)/w = ",
          ((b - tau) + (tau - a) * tmp) / w / (1 - tmp))
    print("1 - tmp =  ", 1 - tmp)
    print("tmp = ", tmp)
    print("ans = ", ans)
    return ans / R


def calc_slope():
    for i in range(8, 15, 1):
        pw = 10**(-i)
        print("is sym? ", Ratio(pw) - Ratio(-pw))

        x0 = pw
        x1 = pw / 10
        y0 = Ratio(x0)
        y1 = Ratio(x1)

        slope = (y0 - y1) / (x0 - x1)
        print(i, slope)


def sim_current():
    N = n + 1  # 节点总数，包括 V0

    V = np.zeros(steps)  # 所有节点电压随时间的矩阵

    I = np.zeros(steps)

    for k in range(1, steps):
        V[k] = V[k - 1]
        t = time[k]

        v_source = np.sin(t * 2 * pi * 1e9)
        V[k] = v_source

        I[k] = C * (V[k] - V[k - 1]) / (2 * dt)


# ------------------ 绘图 ------------------
    plt.figure(figsize=(10, 6))
    """
    for idx in [0, 1, 2]:
        plt.plot(time * 1e9, V[idx], label=f'V{idx}')
    """
    plt.plot(time * 1e9, I, label=f'I')
    plt.xlabel('Time (ns)')
    plt.ylabel('Current (A)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import sympy as sym


def symbol_diff():
    w = sym.symbols('w')
    R = sym.symbols('R')
    C = sym.symbols('C')
    n = sym.symbols('n')
    a = 1 - 1j * R * C * w / 2 + sym.sqrt(-(R * C * w)**2 / 4 - 1j * R * C * w)
    b = 1 - 1j * R * C * w / 2 - sym.sqrt(-(R * C * w)**2 / 4 - 1j * R * C * w)
    tau = 1 - 1j * R * C * w
    t = b / a
    tmp = (tau * b - 1) / (tau * a - 1) * (t**(n - 1))
    ans = ((b - tau) + (tau - a) * tmp) / (1 - tmp)
    y = sym.diff(ans, w)
    print(y)
    print(' ')
    limit_y = sym.limit(y, w, 0)
    print(limit_y)
    limit_y = sym.limit(ans / w, w, 0)
    print(limit_y)
    limit_h = sym.limit(tmp, w, 0)
    print(limit_h)

    sum_a = tau * (a**n) - (a**(n - 1))
    sum_b = tau * (b**n) - (b**(n - 1))
    ano = ((b - tau) * sum_a + (tau - a) * sum_b) / (sum_a - sum_b) / R
    limit_a = sym.limit(ano / w, w, 0)
    print(limit_a)


def symbol_volt():
    w = sym.symbols('w')
    R = sym.symbols('R')
    C = sym.symbols('C')
    t = 1 - 1j * R * C * w
    ans = (t - 1 / t) / (t * t - 1)
    y = sym.diff(ans, w)
    print(y)
    print(' ')
    limit_a = sym.limit(ans, w, 0)
    limit_y = sym.limit(y, w, 0)
    print(limit_a)
    print(limit_y)


def calc_cond():
    N = n + 1  # 节点总数，包括 V0
    G = np.zeros((N, N))  # 电导矩阵
    Cmat = np.zeros((N, N))  # 电容矩阵

    for i in range(N):
        if i > 0:
            G[i, i] += 1 / R
            G[i, i - 1] -= 1 / R
            G[i - 1, i] -= 1 / R
            G[i - 1, i - 1] += 1 / R

    for i in range(N):
        Cmat[i, i] = C
    A = (Cmat / dt + G)
    A[0, :] = 0
    A[0, 0] = 1
    print(A)
    print(np.linalg.cond(A))


def solve_FG():
    lim = 25

    def Y(w):
        if w == 0:
            w = 1e-8
        a = 1 + 1j * R * C * w / 2 + np.sqrt(-(R * C * w)**2 / 4 +
                                             1j * R * C * w)
        b = 1 + 1j * R * C * w / 2 - np.sqrt(-(R * C * w)**2 / 4 +
                                             1j * R * C * w)
        tau = 1 + 1j * R * C * w
        t = b / a
        tmp = (tau * b - 1) / (tau * a - 1) * (t**(n - 1))
        ans = ((tau - b) + (a - tau) * tmp) / (1 - tmp)
        return ans / R

    def Y_div_w(w):
        a = 1 + 1j * R * C * w / 2 + np.sqrt(-(R * C * w)**2 / 4 +
                                             1j * R * C * w)
        b = 1 + 1j * R * C * w / 2 - np.sqrt(-(R * C * w)**2 / 4 +
                                             1j * R * C * w)
        tau = 1 + 1j * R * C * w
        t = b / a
        tmp = (tau * b - 1) / (tau * a - 1) * (t**(n - 1))
        ans = ((tau - b) / w + (a - tau) / w * tmp) / (1 - tmp)
        return ans / R

    def R_V1_V0(w):
        if w == 0:
            w = 1e-8
        tau = 1 + 1j * R * C * w
        ans = 1 / tau
        return ans

    def inte_V1(t):
        return t * np.exp(-t**2 - t + dt)

    def F(w):
        if w == 0:
            w = 1e-8
        ans = (1 + 1j * w * dt - np.exp(1j * w * dt)) / (2 * pi *
                                                         w) * Y_div_w(w)

        ans = -1j / (4 * np.sqrt(pi)) * np.exp(-(w / 2 / M)**2) * (w / M)
        # ans = ans * R_V1_V0(w)
        ans = ans * Y(w)
        ans = ans * np.exp(1j * w * dt)

        return np.real(ans)

    def F_lambda(l):
        if l == 0:
            l = 1e-9 / M
        ans = -1j / (4 * np.sqrt(pi)) * (l**2 * Y_div_w(l * M) *
                                         np.exp(-l**2 / 4) * np.exp(1j * l *
                                                                    (M * dt)))
        return np.real(ans)

    def G(w):
        ans = (-1j / (2 * pi)) * (np.exp(1j * w * dt) - 1) * Y_div_w(w)

        ans = 1 / (2 * np.sqrt(pi)) * np.exp(-(w / 2 / M)**2)
        # ans = ans * R_V1_V0(w)
        ans = ans * Y(w)
        ans = ans * np.exp(1j * w * dt)

        return np.real(ans)

    def G_lambda(l):
        if l == 0:
            l = 1e-9 / M
        ans = 1 / (2 * np.sqrt(pi)) * Y_div_w(l * M) * l * M * np.exp(
            -l**2 / 4) * np.exp(1j * l * (M * dt))
        return np.real(ans)

    for i in range(1, 11):
        n = i

        Fn, Ferr = quad(F_lambda, -lim, lim)
        Gn, Gerr = quad(G_lambda, -lim, lim)
        print(n, "&", f"{Fn:.4e}", "&", f"{Ferr:.4e}", "&", f"{Gn:.4e}", "&",
              f"{Gerr:.4e}", "\\\\")
        #print(n, "&", Fn, "&", Ferr, "&", Gn, "&", Gerr, "\\\\")

    x = np.linspace(-lim, lim, 500000)
    y = np.zeros_like(x)
    for i in range(len(x)):
        y[i] = F_lambda(x[i])

    # ------------------ 绘图 ------------------
    plt.figure(figsize=(8, 5))
    plt.plot(x, y)
    plt.xlabel("w")
    plt.ylabel("Fn(w)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


solve_FG()
