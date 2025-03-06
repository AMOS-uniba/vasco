import numpy as np
import matplotlib as mpl

from abc import abstractmethod
from matplotlib.ticker import MultipleLocator, FuncFormatter

from plots.base import BasePlot


class BaseErrorPlot(BasePlot):
    cmap_dots = mpl.pyplot.get_cmap('autumn_r')
    cmap_meteor = mpl.pyplot.get_cmap('Blues')
    y_formatter = FuncFormatter(lambda x, pos: f'{x:+.2f}')

    intent: str = "position dependent errors"
    target: str

    def __init__(self, widget, **kwargs):
        self.axis_alt = None
        self.axis_az = None
        self.scatter_dots_alt = None
        self.scatter_dots_az = None
        self.scatter_meteor_alt = None
        self.scatter_meteor_az = None
        self.valid_dots = False
        self.valid_meteor = False
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

        self.scatter_dots_alt = self.axis_alt.scatter([], [], s=[], marker='x', c='cyan', zorder=1)
        self.scatter_dots_az = self.axis_az.scatter([], [], s=[], marker='x', c='cyan', zorder=1)
        self.scatter_meteor_alt = self.axis_alt.scatter([], [], s=[], marker='o', c='yellow', linewidth=0.2, zorder=0)
        self.scatter_meteor_az = self.axis_az.scatter([], [], s=[], marker='o', c='yellow', linewidth=0.2, zorder=0)

        self.axis_alt.set_ylim([0, None])
        self.axis_az.set_ylim([0, None])
        self.invalidate()

    def invalidate(self):
        self.invalidate_dots()
        self.invalidate_meteor()

    def invalidate_dots(self):
        self.valid_dots = False

    def invalidate_meteor(self):
        self.valid_meteor = False

    @abstractmethod
    def norm(self, limit):
        """ Create the norm object for the data """

    @abstractmethod
    def set_limits(self, errors):
        """ Set y-axis limits from the data """

    def _update_scatter(self, scatter_alt, scatter_az, positions, magnitudes, errors,
                        *, cmap, limit: float = 1, use_extent: bool = False):
        """ Common method for drawing error data to a scatter """
        assert magnitudes.shape == errors.shape, \
            f"Magnitudes and errors must have the same shape, but are {magnitudes.shape} and {errors.shape}"

        alt = np.degrees(positions[:, 0])
        az = np.degrees(positions[:, 1])

        if errors.size > 0:
            if use_extent:
                self.set_limits(errors)
            scatter_alt.set_offsets(np.stack((alt, errors), axis=1))
            scatter_az.set_offsets(np.stack((az, errors), axis=1))

            norm = self.norm(limit)
            scatter_alt.set_facecolors(cmap(norm(errors)))
            scatter_alt.set_sizes(0.05 * magnitudes)
            scatter_az.set_facecolors(cmap(norm(errors)))
            scatter_az.set_sizes(0.05 * magnitudes)

        self.draw()

    def update_dots(self, positions, magnitudes, errors, *, limit: float = 1):
        self._update_scatter(self.scatter_dots_alt, self.scatter_dots_az, positions, magnitudes, errors,
                             cmap=self.cmap_dots, limit=limit, use_extent=True)
        self.valid_dots = True

    def update_meteor(self, positions, magnitudes, errors, *, limit: float = 1):
        self._update_scatter(self.scatter_meteor_alt, self.scatter_meteor_az, positions, magnitudes, errors,
                             cmap=self.cmap_meteor, limit=limit, use_extent=False)
        self.valid_meteor = True
