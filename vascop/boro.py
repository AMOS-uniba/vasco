#!/usr/bin/env python

import numpy as np
import scipy as sp
import matplotlib as mpl
from matplotlib import pyplot as plt

from typing import Tuple

from shifters import OpticalAxisShifter, EllipticShifter
from transformers import LinearTransformer, ExponentialTransformer, BiexponentialTransformer
from projections import BorovickaProjection

mpl.use('Agg')


COUNT = 100

def plot(i, f, x, y, bx, by, error):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot()
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.scatter(x, y, color='blue', s=100, marker='x')
    ax.scatter(bx, by, color='red', s=50, marker='o')
    ax.quiver(x, y, bx - x, by - y, scale=2)
    ax.set_title(error)
    plt.savefig(f'output/boro-{f:02d}.png', dpi=200)
    plt.close('all')


x = np.random.normal(0, 0.3, size=COUNT)
y = np.random.normal(0, 0.3, size=COUNT)
boro_master = BorovickaProjection(a0=0, x0=0, y0=0, A=0, F=0, V=1.0001, S=0.00677, D=0.0953, P=2.20e-6, Q=0.00638, epsilon=0, E=0)
boro_test = BorovickaProjection(a0=0, x0=0.1, y0=0, A=0, F=0, V=1, S=0.00677, D=0.0953, P=2e-4, Q=0.00638, epsilon=0, E=0)
boro_ident = BorovickaProjection()

s = OpticalAxisShifter(0.3, 5.1, 0.3, 1.1)
r, b = s(x, y)
bx, by = s.inverse(r, b)

f = 0
count = 21
for i in np.linspace(-2, 2, count):
    boro_test = BorovickaProjection(a0=0, x0=0, y0=0, epsilon=0.1, E=i)
    z, a = boro_test(x, y)
    bx, by = z * np.cos(a), z * np.sin(a)
    error = np.sum(np.square(bx - x) + np.square(by - y))
    plot(i, f, x, y, bx, by, error)
    print(f"[{f}] ", end='' if f < count - 1 else '\n', flush=True)
    f += 1
