#!/usr/bin/env python

import numpy as np
from physfields import ScalarField, VectorField, Zernike, ZernikeVector

import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.use("Agg")


def main():
    coefs = np.zeros((55,), dtype=np.float64)
    zernikes = [Zernike(i, j, masked=True) for i in range(0, 10) for j in range(-i, i+2, 2)]
    x = np.linspace(-1, 1, 201)
    y = np.linspace(-1, 1, 201)
    xx, yy = np.meshgrid(x, y)

    for index in range(0, 100):
        coefs += np.random.normal(0, 0.03, size=55)
        coefs *= 0.99
        print(coefs)
        zs = np.ma.masked_where(xx*xx + yy*yy > 1, np.sum([z(xx, yy) for z in zernikes * coefs], axis=0))
        plt.imshow(zs, norm=mpl.colors.TwoSlopeNorm(0))
        plt.savefig(f"{index:3d}.png")
        plt.close('all')
        print(f"[{index:3d}]", sep=" ")


main()
