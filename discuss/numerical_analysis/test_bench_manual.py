import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.cm as cm

import subprocess
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline

pi = np.acos(-1)


def gen_net(s,
            T,
            R,
            C,
            n,
            E,
            filename="circuit.net",
            outputfile="I_V0_output.dat"):
    """
    生成一个 Ngspice .net 文件，包含串联 R-C 网络和电压源，记录电流 I(V0)

    参数:
    - s: 仿真步长（如 1e-12）
    - T: 仿真总时间（如 1e-9）
    - R: 每个电阻的电阻值（单位任意）
    - C: 每个电容的电容值（单位任意）
    - n: 串联单元个数（正整数）
    - E: 电压源幅值
    - filename: 输出 netlist 文件名
    """

    max_step = "1p"
    if s == 1e-9:
        max_step = "1n"

    with open(filename, "w") as f:
        # 电压源
        f.write(f"* with {s} {T} {R} {C} {n} {E}\n")
        f.write(f".OPTIONS SPARSE\n")
        f.write(f"V0 V0 0 {E}\n")

        # 电阻和电容串联
        for i in range(1, n + 1):
            f.write(f"R{i} V{i-1} V{i} {R}\n")
        for i in range(0, n + 1):
            f.write(f"C{i} V{i} 0 {C}\n")

        # 仿真时间设置
        if s == 1e-12 and T == 1e-9:
            f.write(".tran 1p 1n\n")
        elif s == 1e-9 and T == 1e-7:
            f.write(".tran 1n 100n\n")
        else:
            # 默认写入传入的 s 和 T
            f.write(f".tran {s} {T}\n")

        # 控制块
        f.write(".control\n")
        f.write("run\n")
        f.write(f"wrdata {outputfile} I(V0)\n")
        f.write("quit\n")
        f.write(".endc\n")

        f.write(".end\n")


def cmp(s, T, R, C, n, E):
    ref_input = "circuit_std.net"
    ref_output = "I_V0_output_std.dat"
    gen_net(s, T, R, C, n, E, filename=ref_input, outputfile=ref_output)
    subprocess.run(["ngspice", "-b", ref_input],
                   check=True,
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

    def read_data(filename):
        data = np.loadtxt(filename, comments=['*', '$'])  # 跳过注释行
        if data.ndim == 1:
            data = data.reshape(1, -1)
        t, i = data[:, 0], data[:, 1]
        return t, i

    t_ref, i_ref_tmp = read_data(ref_output)
    intef = CubicSpline(t_ref, -i_ref_tmp, bc_type='natural')

    # interp1d(t_ref, -i_ref_tmp, kind='linear', fill_value='extrapolate')

    def f(k):
        t = k * s
        if "SIN" in E:
            if "1G" in E:
                l = 2 * pi * 1e9
            elif "100MEG" in E:
                l = 2 * pi * 1e8
            else:
                print(E)
                assert False
            if "90" in E:
                return np.cos(l * t) * (
                    (l * s)**2 / 2) - np.sin(l * t) * np.sin(l * s)
            else:
                return np.sin(l * t) * (
                    (l * s)**2 / 2) + np.cos(l * t) * np.sin(l * s)
        else:
            assert False

    def f2(k):
        t = k * s
        if "SIN" in E:
            if "1G" in E:
                l = 2 * pi * 1e9
            elif "100MEG" in E:
                l = 2 * pi * 1e8
            assert "90" in E
            return -l * l * np.cos(l * t)

    if s == 1e-12:
        steps = 1001
    if s == 1e-9:
        steps = 101
    t_step = np.linspace(0, T, steps)

    i_sim = np.zeros(steps)
    i_ref = np.zeros(steps)

    N = n + 1
    G = sp.lil_matrix((N, N))
    Cmat = sp.lil_matrix((N, N))

    for i in range(N):
        if i > 0:
            G[i, i] += 1 / R
            G[i, i - 1] -= 1 / R
            G[i - 1, i] -= 1 / R
            G[i - 1, i - 1] += 1 / R

    for i in range(N):
        Cmat[i, i] = C

    G = G.tocsr()
    Cmat = Cmat.tocsr()
    A = (Cmat / s + G)
    A[0, :] = 0
    A[0, 0] = 1
    A = A.tocsc()
    solver = spla.factorized(A)

    V = np.zeros((N, steps))
    i_ref_my = np.zeros(steps)
    V[:, 0] = 0

    for k in range(1, steps):
        t = t_step[k]
        b = np.zeros(N)
        b = -G @ V[:, k - 1]
        b[0] = f(k)
        dV = solver(b)
        V[:, k] = V[:, k - 1] + dV
        i_ref_my[k] = C * np.sum(dV / s)

        i_ref[k] = intef(t)

        # when C ~ 1e-15
        assert C == 1e-15
        i_sim[k] = (n + 1) * C * b[0] / s - n * (n + 1) / 2 * R * C * C * f2(k)
        print((n + 1) * C * b[0] / s, n * (n + 1) / 2 * R * C * C * f2(k))
        #i_sim[k] = 1.2899991765973618e-13 * b[0] / s

    sim_output = "I_V0_output.dat"
    ref_output = "I_V0_output_std_intef.dat"
    ref_my_output = "I_V0_output_my.dat"

    with open(sim_output, "w") as f:
        for k in range(0, steps):
            f.write(f"{t_step[k]} {i_sim[k]}\n")
    with open(ref_output, "w") as f:
        for k in range(0, steps):
            f.write(f"{t_step[k]} {i_ref[k]}\n")
    with open(ref_my_output, "w") as f:
        for k in range(0, steps):
            f.write(f"{t_step[k]} {i_ref_my[k]}\n")

    # Step 5: 误差计算
    abs_err = np.abs(i_sim - i_ref)
    rel_err = abs_err / (np.abs(i_ref) + np.abs(i_sim) + 1e-20)  # 避免除 0

    abs_err_my = np.abs(i_ref_my - i_sim)
    rel_err_my = abs_err_my / (np.abs(i_ref_my) + np.abs(i_sim) + 1e-20
                               )  # 避免除 0

    abs_err_ref = np.abs(i_ref_my - i_ref)
    rel_err_ref = abs_err_ref / (np.abs(i_ref_my) + np.abs(i_ref) + 1e-20)

    mae = np.mean(abs_err)
    mae_my = np.mean(abs_err_my)
    mae_ref = np.mean(abs_err_ref)

    max_ae = np.max(abs_err)

    mre = np.mean(rel_err)
    mre_my = np.mean(rel_err_my)
    mre_ref = np.mean(rel_err_ref)

    max_re = np.max(rel_err)

    # 输出结果
    print(f"最大绝对误差 (Max AE): {max_ae:.4e}")
    print(f"最大相对误差 (Max RE): {max_re:.4e}")
    min_m = min(mae, mre)
    min_max = min(max_ae, max_re)
    print("mae = ", mae, " mre = ", mre)
    print("mae_my = ", mae_my, " mre_my = ", mre_my)
    print("mae_ref = ", mae_ref, " mre_ref = ", mre_ref)

    return mae, mre, mae_my, mre_my, mae_ref, mre_ref


s_list = [1e-12, 1e-9]
T_list = [1e-9, 1e-7]
freq_list = [10 / T_list[0], 10 / T_list[1]]

R_list = np.linspace(0.1, 10, 200)
R_list = [1]
C_list = np.linspace(0.1, 10, 200)
C_list = [1e-15]

n_list = [i + 1 for i in range(128)]

E_list = [[
    "SIN(0 1 1G 0 0 90)", "PULSE(-1 1 2PS 20PS 20PS 500PS 1NS)",
    "EXP(-4 -1 20PS 300PS 600PS 400PS)"
],
          [
              "SIN(0 1 100MEG 0 0 90)", "PULSE(-1 1 2NS 2NS 2NS 50NS 100NS)",
              "EXP(-4 -1 2NS 30NS 60NS 40NS)"
          ]]

#s_list = [1e-12]
#T_list = [1e-9]
#n_list = [128]

data = []

for i in range(len(s_list)):
    s = s_list[i]
    T = T_list[i]
    for R in R_list:
        for C in C_list:
            for E in E_list[i]:
                if "SIN" not in E:
                    break
                for n in n_list:
                    print("params = ", s, T, R, C, n, E)
                    mae, mre, mae_my, mre_my, mae_ref, mre_ref = cmp(
                        s, T, R, C, n, E)
                    data.append([
                        s, T, R, C, E, n, mae, mre, mae_my, mre_my, mae_ref,
                        mre_ref
                    ])

import os
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams.update({
    #"text.usetex": True,
    #"font.family": "serif",
    "font.size": 10,
    "figure.figsize": (4, 3),
    "axes.grid": True,
    "grid.linestyle": ":",
    "legend.frameon": False,
    "savefig.bbox": "tight"
})


def plot_errors(data, output_dir="pic"):
    """
    data: list of [s, T, R, C, E, n, abs_err, rel_err, abs_err_my, rel_err_my, abs_err_ref, rel_err_ref]
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 分组：key = (s, T, R, C, E)
    grouped = defaultdict(list)
    for row in data:
        key = tuple(row[:5])  # (s, T, R, C, E)
        grouped[key].append(row)

    for key, rows in grouped.items():
        s, T, R, C, E = key
        rows.sort(key=lambda x: x[5])  # 按 n 排序

        n_vals = [r[5] for r in rows]
        abs_errs = [r[6] for r in rows]
        rel_errs = [r[7] for r in rows]
        abs_errs_my = [r[8] for r in rows]
        rel_errs_my = [r[9] for r in rows]
        abs_errs_ref = [r[10] for r in rows]
        rel_errs_ref = [r[11] for r in rows]

        fig, ax = plt.subplots()
        ax.plot(n_vals, abs_errs, '-', label=r'Abs Err (vs. Ng)')
        ax.plot(n_vals, rel_errs, '--', label=r'Rel Err (vs. Ng)')
        ax.plot(n_vals, abs_errs_my, '-', label=r'Abs Err (vs. Py)')
        ax.plot(n_vals, rel_errs_my, '--', label=r'Rel Err (vs. Py)')
        ax.plot(n_vals, abs_errs_ref, '-', label=r'Abs Err (Ng vs. Py)')
        ax.plot(n_vals, rel_errs_ref, '--', label=r'Rel Err (Ng vs. Py)')

        #ax.set_yscale('log')
        ax.set_xlabel(r'$n$')
        ax.set_ylabel(r'Error')
        ax.set_title(rf"$s={s:.0e},\ T={T:.0e},\ R={R},\ C={C},\ E={E}$")
        ax.legend()
        filename = f"s={s:.0e}_T={T:.0e}_R={R}_C={C}_E={E}.pdf"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    print(f"Saved {len(grouped)} plots to {output_dir}/")


plot_errors(data)
