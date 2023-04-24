#!/usr/bin/env python

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt

import argparsedirs
from utilities import by_azimuth, polar_to_cart


from correctors import Interpolator, ZernikeExpander, ZernikeDiscreteExpander, KernelSmoother, kernels

mpl.use('Agg')
mpl.rc('font', family='Minion Pro')
np.set_printoptions(edgeitems=999, linewidth=100000, formatter={'float': lambda x: f"{x:8.6f}"})
RADIUS = 1
RESOLUTION = 50


class Corrector():
    METHODS = ['zernike', 'interpolate', 'interpolate-cubic']

    quiver_options = dict(
        pivot='middle',
        color='black',
        scale=0.05,
        width=0.002,
    )

    def __init__(self):
        self.argparser = argparse.ArgumentParser("Virtual corrector plate")
        self.argparser.add_argument('infile', type=argparse.FileType('r'), help="input file")
        self.argparser.add_argument('outdir', action=argparsedirs.WriteableDir, help="output directory")
        self.argparser.add_argument('method', type=str, choices=Corrector.METHODS)
        self.args = self.argparser.parse_args()
        self.outdir = Path(self.args.outdir)

    def load(self):
        df = pd.read_csv(self.args.infile.name, sep='\t', header=0, nrows=500)
        df['a_cat_rad'] = np.radians(df['acat'])
        df['z_cat_rad'] = df['zcat'] / 90
        df['x_cat'], df['y_cat'] = polar_to_cart(df['z_cat_rad'], df['a_cat_rad'])
        df['a_com_rad'] = np.radians(df['acom'])
        df['z_com_rad'] = df['zcom'] / 90
        df['x_com'], df['y_com'] = polar_to_cart(df['z_com_rad'], df['a_com_rad'])
        df['dx'], df['dy'] = df['x_cat'] - df['x_com'], df['y_cat'] - df['y_com']
        self.data = df
        self.points = self.data[['x_cat', 'y_cat']].to_numpy()
        self.values = self.data[['dx', 'dy']].to_numpy()

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

    def plot_raw_data(self, axes):
        axes.scatter(self.points[:, 0], self.points[:, 1], s=3, color='blue')
        axes.quiver(self.points[:, 0], self.points[:, 1],
                    self.values[:, 0], self.values[:, 1],
                    width=0.0025, color='red')

    def plot_data(self, filename):
        fig, ax_data, ax_grid = self.empty_plot()
        self.plot_raw_data(ax_data)
        fig.savefig(self.outdir / filename, dpi=200)

    def interpolate(self, nodes):
        interpolator = Interpolator(self.points, self.values, method='cubic')
        return interpolator(nodes)

    def interpolate_twostep(self, nodes):
        interpolator = Interpolator(self.points, self.values, method='cubic')
        convex = interpolator(nodes)
        newstacked = nodes[~np.isnan(convex).any(axis=1), :]
        convex = convex[~np.isnan(convex).any(axis=1), :]
        extrapolator = Interpolator(newstacked, convex, method='nearest')
        result = np.ma.masked_array(extrapolator(nodes), nodes.mask)
        return result

    def kernel_smooth(self, nodes, **kwargs):
        smoother = KernelSmoother(self.points, self.values, **kwargs)
        return smoother(nodes)

    def zernike_expand(self, nodes, **kwargs):
        expander = ZernikeExpander(self.points, self.values, **kwargs)
        return expander(nodes, order=kwargs.get('order', 11))

    def prepare_grid(self, resolution):
        x = np.linspace(-RADIUS, RADIUS, resolution)
        y = np.linspace(-RADIUS, RADIUS, resolution)
        xx, yy = np.meshgrid(x, y)
        xx = np.ma.masked_where(xx**2 + yy**2 > 1, xx)
        yy = np.ma.masked_where(xx**2 + yy**2 > 1, yy)
        return xx, yy

    def plot_magnitude(self, ax, uv):
        extent = (RESOLUTION + 1) / RESOLUTION
        ax.imshow(
            by_azimuth(uv),
            extent=[-extent, extent, -extent, extent],
            origin='lower',
            interpolation='antialiased',
            alpha=0.5,
        )

    def plot(self, method, filename, **options):
        fig, ax_data, ax_grid = self.empty_plot()

        xx, yy = self.prepare_grid(RESOLUTION)
        disk = xx**2 + yy**2 > 0.98
        nodes = np.ma.stack((xx.ravel(), yy.ravel()), axis=1)
        uv = method(nodes, **options).reshape((RESOLUTION, RESOLUTION, 2))
        uv = np.ma.masked_where(np.stack([disk, disk], axis=2), uv)
        ax_data.quiver(xx, yy, uv[:, :, 0], uv[:, :, 1], **self.quiver_options)

        xx, yy = self.prepare_grid(200)
        disk = xx**2 + yy**2 > 0.98
        nodes = np.ma.stack((xx.ravel(), yy.ravel()), axis=1)
        uv = method(nodes, **options).reshape((200, 200, 2))
        uv = np.ma.masked_where(np.stack([disk, disk], axis=2), uv)
        self.plot_magnitude(ax_data, uv)

        self.plot_raw_data(ax_data)
        fig.savefig(self.outdir / filename, dpi=200)
        plt.close('all')
        print(f"Plotting to {self.outdir / filename}")


corr = Corrector()
corr.load()
# corr.generate()
corr.plot_data('raw.pdf')
corr.plot(corr.interpolate, 'interpolated.pdf')
corr.plot(corr.interpolate_twostep, 'interpolated-nearest.pdf')
corr.plot(corr.kernel_smooth, 'kernel-smoothed.pdf', kernel=kernels.nexp, bandwidth=0.05)
corr.plot(corr.zernike_expand, 'zerniked.pdf', kernel=kernels.nexp, bandwidth=0.05, order=19)

# for i in np.arange(-20, 11):
#    bw = 10**(i / 10)
#    corr.plot_kernel_smoothed(f'kernel-smoothed-{i + 20:02d}.png', kernel=kernels.nexp, bandwidth=bw)
#    print(i, bw)
