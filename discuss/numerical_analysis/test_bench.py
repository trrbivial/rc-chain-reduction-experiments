import numpy as np
import matplotlib.cm as cm
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.stats import gmean

large_C = 1e-7


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


import subprocess


def cmp(s,
        T,
        R,
        C,
        n,
        E,
        sim_input="circuit.net",
        ref_input="circuit_std.net",
        sim_output="I_V0_output.dat",
        ref_output="I_V0_output_std.dat"):
    """
    生成 netlist，运行 Ngspice，比较仿真输出与参考数据。
    """

    shared_C = C
    if C == 1e-15:
        l = int(np.ceil(np.log2(n + 1)))
        l = 1
        if n + 1 > 64:
            l = (n + 1) // 2
        shared_C = C * (n + 1) / l
    elif C == large_C:
        l = 2
        shared_C = C
    else:
        assert False
    gen_net(s, T, R, C, n, E, filename=ref_input, outputfile=ref_output)
    subprocess.run(["ngspice", "-b", ref_input],
                   check=True,
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

    gen_net(s,
            T,
            R,
            shared_C,
            l - 1,
            E,
            filename=sim_input,
            outputfile=sim_output)
    subprocess.run(["ngspice", "-b", sim_input],
                   check=True,
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

    # Step 3: 读取仿真输出文件
    def read_data(filename):
        data = np.loadtxt(filename, comments=['*', '$'])  # 跳过注释行
        if data.ndim == 1:
            data = data.reshape(1, -1)
        t, i = data[:, 0], data[:, 1]
        return t, i

    t_sim, i_sim_tmp = read_data(sim_output)
    f_sim = interp1d(t_sim,
                     -i_sim_tmp,
                     kind='linear',
                     fill_value='extrapolate')

    t_ref, i_ref_tmp = read_data(ref_output)
    f_ref = interp1d(t_ref,
                     -i_ref_tmp,
                     kind='linear',
                     fill_value='extrapolate')

    steps = int(np.ceil(T / s)) + 1
    tstep = np.linspace(0, T, steps)

    i_sim = np.zeros(steps)
    i_ref = np.zeros(steps)
    for k in range(1, steps):
        t = tstep[k]
        i_sim[k] = f_sim(t)
        i_ref[k] = f_ref(t)

    # Step 5: 误差计算
    abs_err = np.abs(i_sim - i_ref)
    rel_err = abs_err / (np.abs(i_sim) + np.abs(i_ref) + 1e-20)  # 避免除 0

    for i in range(len(rel_err)):
        rel = rel_err[i]
        #if rel > 0.2:
        #print(i, rel, i_sim[i], i_ref[i])

    w = np.abs(i_sim) + np.abs(i_ref)

    i_sum = np.sum(w)
    mae = np.mean(abs_err)
    mre = np.sum(w * rel_err) / i_sum

    #mae = np.mean(abs_err)
    #max_ae = np.max(abs_err)
    #mre = np.mean(rel_err)
    #max_re = np.max(rel_err)

    # 输出结果
    #print(f"最大绝对误差 (Max AE): {max_ae:.4e}")
    #print(f"最大相对误差 (Max RE): {max_re:.4e}")
    print("mae = ", mae, " mre = ", mre)

    return mae, mre


s_list = [1e-12, 1e-9]
T_list = [1e-9, 1e-7]
freq_list = [10 / T_list[0], 10 / T_list[1]]

R_list = np.linspace(0.1, 10, 200)
R_list = [1]
C_list = np.linspace(0.1, 10, 200)
C_list = [1e-15, large_C]
n_list = [i + 1 for i in range(128)]

#n_list = [128]

E_list = [[
    "SIN(0 1 1G 0 0 90)", "PULSE(-1 1 2PS 200PS 200PS 500PS 1NS)",
    "EXP(-4 -1 20PS 300PS 600PS 400PS)"
],
          [
              "SIN(0 1 100MEG 0 0 90)", "PULSE(-1 1 2NS 2NS 2NS 50NS 100NS)",
              "EXP(-4 -1 2NS 30NS 60NS 40NS)"
          ]]

data = []

for R in R_list:
    for C in C_list:
        for eid in range(3):
            for n in n_list:
                mae = np.zeros(2)
                mre = np.zeros(2)
                for i in range(2):
                    s = s_list[i]
                    T = T_list[i]
                    E = E_list[i][eid]
                    print("params = ", s, T, R, C, n, E)
                    mae[i], mre[i] = cmp(s, T, R, C, n, E)
                data.append([R, C, E_list[0][eid][0:3], n, mae, mre])

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

    blues = [cm.Blues(x) for x in np.linspace(0.3, 0.9, 2)]
    blacks = [cm.Greens(x) for x in np.linspace(0.3, 0.9, 2)]

    # 合并为一个颜色列表（每个是 RGBA，可用于 matplotlib）
    colors = blues + blacks
    """
    data: list of [R, C, E, n, abs_err, rel_err]
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 分组：key = (R, C, E)
    grouped = defaultdict(list)
    for row in data:
        key = tuple(row[:3])  # (R, C, E)
        grouped[key].append(row)

    for key, rows in grouped.items():
        R, C, E = key
        rows.sort(key=lambda x: x[3])  # 按 n 排序

        n_vals = [r[3] for r in rows]
        abs_errs = [[r[4][0] for r in rows], [r[4][1] for r in rows]]
        rel_errs = [[r[5][0] for r in rows], [r[5][1] for r in rows]]

        fig, ax = plt.subplots()
        ax.plot(n_vals,
                abs_errs[0],
                '-',
                label=r'Abs Err (1ps/1ns)',
                color=colors[0])
        ax.plot(n_vals,
                rel_errs[0],
                '--',
                label=r'Rel Err (1ps/1ns)',
                color=colors[1])
        ax.plot(n_vals,
                abs_errs[1],
                '-',
                label=r'Abs Err (1ns/100ns)',
                color=colors[2])
        ax.plot(n_vals,
                rel_errs[1],
                '--',
                label=r'Rel Err (1ns/100ns)',
                color=colors[3])

        if C == 1e-15:
            ax.set_yscale('log')
        elif C == large_C:
            ax.set_yscale('log')
        else:
            assert False

        ax.set_xlabel(r'$n$')
        ax.set_ylabel(r'Error')
        ax.set_title(rf"R={R}, C={C}, E={E}")
        ax.legend()
        filename = f"R={R}_C={C}_E={E}.pdf"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    print(f"Saved {len(grouped)} plots to {output_dir}/")


plot_errors(data)
