import numpy as np
import matplotlib as mpl

from matplotlib.ticker import MultipleLocator

from .base import BasePlot
from matchers import Matcher


class ErrorPlot(BasePlot):
    def __init__(self, widget, **kwargs):
        self.cmap = mpl.cm.get_cmap('autumn_r')
        super().__init__(widget, **kwargs)

    def add_axes(self):
        self.axisAlt = self.figure.add_subplot(2, 1, 1)
        self.axisAz  = self.figure.add_subplot(2, 1, 2)

        self.axisAlt.set_xlim([0, 90])
        self.axisAlt.set_ylim([0, None])
        self.axisAlt.set_xlabel('zenith distance')
        self.axisAlt.xaxis.set_major_locator(MultipleLocator(10))
        self.axisAlt.xaxis.set_major_formatter(lambda x, pos: f'{x:.0f}°')
        self.axisAlt.set_ylabel('error')
        self.axisAlt.yaxis.set_major_formatter(lambda x, pos: f'{x:.2f}°')
        self.axisAlt.grid(color='white', alpha=0.2)

        self.axisAz.set_xlim([0, 360])
        self.axisAz.set_ylim([0, None])
        self.axisAz.set_xlabel('azimuth')
        self.axisAz.xaxis.set_major_locator(MultipleLocator(45))
        self.axisAz.xaxis.set_major_formatter(lambda x, pos: f'{x:.0f}°')
        self.axisAz.set_ylabel('error')
        self.axisAz.yaxis.set_major_formatter(lambda x, pos: f'{x:.2f}°')
        self.axisAz.grid(color='white', alpha=0.2)

        self.scatterAlt = self.axisAlt.scatter([], [], s=[], marker='x', c='cyan')
        self.scatterAz = self.axisAz.scatter([0], [0], s=[1], marker='x', c='cyan')
        self.invalidate()

    def invalidate(self):
        self.valid = False

    def update(self, positions, magnitudes, errors, *, limit=1):
        alt = np.degrees(positions[:, 0])
        az = np.degrees(positions[:, 1])

        max_error = Matcher.max_error(errors)
        avg_error = Matcher.avg_error(errors)

        if max_error is not np.nan:
            self.axisAlt.set_ylim([0, np.degrees(max_error) * 1.05])
            self.axisAz.set_ylim([0, np.degrees(max_error) * 1.05])
            self.scatterAlt.set_offsets(np.stack((alt, np.degrees(errors)), axis=1))
            self.scatterAz.set_offsets(np.stack((az, np.degrees(errors)), axis=1))

            norm = mpl.colors.Normalize(vmin=0, vmax=limit)
            self.scatterAlt.set_facecolors(self.cmap(norm(errors)))
            self.scatterAlt.set_sizes(0.05 * magnitudes)
            self.scatterAz.set_facecolors(self.cmap(norm(errors)))
            self.scatterAz.set_sizes(0.05 * magnitudes)

        self.draw()
