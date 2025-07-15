import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import sympy as sym
from sympy import oo


def symbol_inte():
    t = sym.symbols('t')
    w = sym.symbols('w')
    R = sym.symbols('R')
    C = sym.symbols('C')
    n = sym.symbols('n')

    R = 1
    C = 2

    a = 1 + 1j * R * C * w / 2 + sym.sqrt(-(R * C * w)**2 / 4 + 1j * R * C * w)
    b = 1 + 1j * R * C * w / 2 - sym.sqrt(-(R * C * w)**2 / 4 + 1j * R * C * w)
    tau = 1 + 1j * R * C * w
    t = b / a
    tmp = (tau * b - 1) / (tau * a - 1) * (t**(n - 1))
    ans = ((tau - b) + (a - tau) * tmp) / (1 - tmp)
    ans = ans / R

    limit_y = sym.limit(ans, w, 0)
    print("ans(0) = ", limit_y)

    limit_y = sym.limit(ans, w, oo)
    limit_y = sym.latex(limit_y)
    print("ans(oo) = ", limit_y)

    limit_y = sym.limit(ans / w, w, 0)
    print("ans/w(0) = ", limit_y)

    limit_y = sym.limit(ans / w, w, oo)
    limit_y = sym.latex(limit_y)
    print("ans/w(oo) = ", limit_y)

    limit_h = sym.limit(tmp, w, 0)
    print("h(0) = ", limit_h)

    limit_h = sym.limit(tmp, w, oo)
    limit_h = sym.latex(limit_h)
    print("h(oo) = ", limit_h)

    sum_a = tau * (a**n) - (a**(n - 1))
    sum_b = tau * (b**n) - (b**(n - 1))
    ano = ((b - tau) * sum_a + (tau - a) * sum_b) / (sum_a - sum_b) / R
    limit_a = sym.limit(ano / w, w, 0)
    print(limit_a)


symbol_inte()
