#!/usr/bin/env python

import numpy as np
from correctors import Interpolator, ZernikeExpander, ZernikeFitter, KernelSmoother, kernels
from physfields import ZernikeVector, VectorField

import matplotlib as mpl
from matplotlib import pyplot as plt
from utilities import by_azimuth
from logger import setupLog

log = setupLog(__name__)

RADIUS = 1
RESOLUTION = 50

mpl.use('Agg')
mpl.rc('font', family='Minion Pro')


class Expander:
    quiver_options = dict(
        pivot='middle',
        color='black',
#        scale=2,
        width=0.002,
    )

    def prepare_grid(self):
        x = np.linspace(-RADIUS, RADIUS, RESOLUTION)
        y = np.linspace(-RADIUS, RADIUS, RESOLUTION)
        xx, yy = np.meshgrid(x, y)
        xx = np.ma.masked_where(xx**2 + yy**2 > 1, xx)
        yy = np.ma.masked_where(xx**2 + yy**2 > 1, yy)
        return xx, yy

    def expand(self, field, xx, yy, order):
        points = np.stack((xx.ravel(), yy.ravel()), axis=1)

        u, v = field(xx, yy)
        uv = np.ma.stack((u.ravel(), v.ravel()), axis=1)
        self.uv = (u, v)
        return ZernikeFitter()(points, uv, order)

    def empty_plot(self):
        fig = plt.figure(figsize=(5, 5))
        ax_data = fig.add_subplot()
        ax_grid = fig.add_subplot(polar=True, frameon=False)
        ax_grid.set_xticklabels([])
        ax_grid.set_yticklabels([])
        ax_grid.grid(True, antialiased=True, color='gray', linewidth=0.1)
        ax_data.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
        ax_data.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
        ax_data.set_xlim([-1, 1])
        ax_data.set_ylim([-1, 1])
        ax_data.set_aspect('equal')
        fig.tight_layout()
        circle = plt.Circle((0, 0), 1, color='black', linewidth=5, fill=False)
        ax_data.add_collection(mpl.collections.PatchCollection([circle], zorder=-10, match_original=True))
        return fig, ax_data, ax_grid

    def plot_magnitude(self, ax, uv):
        extent = (RESOLUTION + 1) / RESOLUTION
        ax.imshow(
            by_azimuth(uv),
            extent=[-extent, extent, -extent, extent],
            origin='lower',
            interpolation='antialiased',
            alpha=0.5,
        )

    def plot(self, what, xx, yy, filename, **options):
        fig, ax_data, ax_grid = self.empty_plot()
        disk = xx**2 + yy**2 > 0.99

        uv = np.ma.masked_where(np.ma.stack([disk, disk], axis=2), what.reshape((RESOLUTION, RESOLUTION, -1)))

        self.plot_magnitude(ax_data, uv)
        ax_data.quiver(xx, yy, uv[:, :, 0], uv[:, :, 1], **self.quiver_options)
        fig.savefig(filename, dpi=200)
        plt.close('all')
        log.info(f"Plotting to {filename}")

    def go(self, true, expanded):
        self.field = ZernikeVector.create(4, 0, True)
        self.field = VectorField(lambda x, y: (0.4 * np.sin(2 * (2 * x - y)) * np.exp(-(x / 2)**2), -3 * np.cos(2 * y * x) * (x * x * y)))
        self.field = VectorField(lambda x, y: (np.sin(5 * (x - 2 * y)) * np.exp(-x**2), np.cos(7 * y * x) * (x**2 - 1)))
#        self.field = VectorField(lambda x, y: (y - 0.2*x*y, -x + 3 * np.sin(x*y)))
        xx, yy = self.prepare_grid()

        raw = np.stack(self.field(xx, yy), axis=2)
        self.plot(raw, xx, yy, true)

        for i in range(1, 21):
            expansion = self.expand(self.field, xx, yy, i).reshape(RESOLUTION, RESOLUTION, -1)
            gof = np.sum(np.square(raw - expansion)) / raw.count() * 2
            log.debug(gof)
            self.plot(expansion, xx, yy, expanded.format(i))


e = Expander()
e.go('true.pdf', 'expanded-{:02d}.pdf')
