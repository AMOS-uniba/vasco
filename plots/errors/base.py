import numpy as np
import matplotlib as mpl

from abc import abstractmethod
from matplotlib.ticker import MultipleLocator, FuncFormatter

from plots.base import BasePlot


class BaseErrorPlot(BasePlot):
    cmap = mpl.cm.get_cmap('autumn_r')
    y_formatter = FuncFormatter(lambda x, pos: f'{x:+.2f}')

    def __init__(self, widget, **kwargs):
        self.axis_alt = None
        self.axis_az = None
        self.scatter_alt = None
        self.scatter_az = None
        self.valid = False
        super().__init__(widget, **kwargs)

    def add_axes(self):
        self.axis_alt = self.figure.add_subplot(2, 1, 1)
        self.axis_az = self.figure.add_subplot(2, 1, 2)

        self.axis_alt.set_xlim([0, 90])
        self.axis_alt.set_xlabel('zenith distance')
        self.axis_alt.xaxis.set_major_locator(MultipleLocator(10))
        self.axis_alt.xaxis.set_major_formatter(lambda x, pos: f'{x:.0f}°')
        self.axis_alt.set_ylabel('error')
        self.axis_alt.yaxis.set_major_formatter(self.y_formatter)
        self.axis_alt.grid(color='white', alpha=0.2)

        self.axis_az.set_xlim([0, 360])
        self.axis_az.set_xlabel('azimuth')
        self.axis_az.xaxis.set_major_locator(MultipleLocator(45))
        self.axis_az.xaxis.set_major_formatter(lambda x, pos: f'{x:.0f}°')
        self.axis_az.set_ylabel('error')
        self.axis_az.yaxis.set_major_formatter(self.y_formatter)
        self.axis_az.grid(color='white', alpha=0.2)

        self.scatter_alt = self.axis_alt.scatter([], [], s=[], marker='x', c='cyan')
        self.scatter_az = self.axis_az.scatter([0], [0], s=[1], marker='x', c='cyan')
        self.invalidate()

    def invalidate(self):
        self.valid = False

    @abstractmethod
    def norm(self, limit):
        """ Create the norm object for the data """

    @abstractmethod
    def set_limits(self, errors):
        """ Set y-axis limits from the data """

    def update(self, positions, magnitudes, errors, *, limit=1):
        assert magnitudes.shape == errors.shape,\
            f"Magnitudes and errors should have the same shape, are {magnitudes.shape} and {errors.shape}"

        alt = np.degrees(positions[:, 0])
        az = np.degrees(positions[:, 1])

        self.set_limits(errors)

        if errors.size > 0:
            self.set_limits(errors)
            self.scatter_alt.set_offsets(np.stack((alt, errors), axis=1))
            self.scatter_az.set_offsets(np.stack((az, errors), axis=1))

            norm = self.norm(limit)
            self.scatter_alt.set_facecolors(self.cmap(norm(errors)))
            self.scatter_alt.set_sizes(0.05 * magnitudes)
            self.scatter_az.set_facecolors(self.cmap(norm(errors)))
            self.scatter_az.set_sizes(0.05 * magnitudes)

        self.draw()
