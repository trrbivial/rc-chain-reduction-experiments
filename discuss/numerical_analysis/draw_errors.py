import os
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
    "figure.figsize": (4, 3),
    "axes.grid": True,
    "grid.linestyle": ":",
    "legend.frameon": False,
    "savefig.bbox": "tight"
})


def plot_errors(data, output_dir="pic"):
    """
    data: list of [s, T, R, C, n, E, abs_err, rel_err]
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
        rows.sort(key=lambda x: x[4])  # 按 n 排序

        n_vals = [r[4] for r in rows]
        abs_errs = [r[6] for r in rows]
        rel_errs = [r[7] for r in rows]

        fig, ax = plt.subplots()
        ax.plot(n_vals, abs_errs, 'o-', label=r'Absolute Error')
        ax.plot(n_vals, rel_errs, 's--', label=r'Relative Error')

        ax.set_xlabel(r'$n$')
        ax.set_ylabel(r'Error')
        ax.set_title(rf"$s={s:.0e},\ T={T:.0e},\ R={R},\ C={C},\ E={E}$")
        ax.legend()
        filename = f"s={s:.0e}_T={T:.0e}_R={R}_C={C}_E={E}.pdf".replace(
            '.', 'p')
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    print(f"Saved {len(grouped)} plots to {output_dir}/")
