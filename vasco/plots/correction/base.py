import numpy as np
import matplotlib as mpl

from abc import abstractmethod

from plots.base import BasePlot
from utilities import altaz_to_disk, proj_to_disk, numpy_to_disk


class BaseCorrectionPlot(BasePlot):
    cmap_dots = mpl.pyplot.get_cmap('autumn_r')
    cmap_grid = mpl.pyplot.get_cmap('Greens')
    cmap_meteor = mpl.pyplot.get_cmap('RdYlGn')
    colour_dots = 'white'
    colour_meteor = 'cyan'

    intent: str = "correction plot"
    target: str

    def __init__(self, widget, **kwargs):
        self.scatter_dots = None
        self.scatter_meteor = None
        self.valid_dots = False
        self.valid_grid = False
        self.valid_meteor = False
        super().__init__(widget, **kwargs)

    def add_axes(self):
        self.axis = self.figure.add_subplot()
        self.axis.set_xlim([-1, 1])
        self.axis.set_ylim([-1, 1])
        self.axis.set_xlabel('x')
        self.axis.set_ylabel('y')
        self.axis.set_aspect('equal')
        self.axis.grid(color='white', alpha=0.2)

        self.scatter_dots = self.axis.scatter([], [],
                                              s=[], marker='o', c=self.colour_dots,
                                              linewidth=0.3, edgecolor='black')
        self.scatter_meteor = self.axis.scatter([], [],
                                                s=[], marker='o', c=self.colour_meteor,
                                                linewidth=0.2, edgecolor='black')

    def invalidate_dots(self):
        self.valid_dots = False

    def invalidate_grid(self):
        self.valid_grid = False

    def invalidate_meteor(self):
        self.valid_meteor = False

    def invalidate(self):
        self.invalidate_dots()
        self.invalidate_grid()
        self.invalidate_meteor()

    def update_dots(self, pos_cat, pos_obs, mag_cat, mag_obs, *, limit=1, scale=0.05):
        pos_cat = numpy_to_disk(pos_cat)
        pos_obs = proj_to_disk(pos_obs)

        self._update_dots(pos_cat, pos_obs, mag_cat, mag_obs, limit=limit, scale=scale)
        self.valid_dots = True
        self.draw()

    @abstractmethod
    def _update_dots(self, pos_cat, pos_obs, mag_cat, mag_obs, *, limit, scale):
        """ Inner method for updating the dots """

    def clear_errors(self):
        self.update_dots(None, None, None, None)

    def update_meteor(self, pos_obs, pos_corr, mag_obs, mag_corr, *, scale=0.05):
        pos_obs = proj_to_disk(pos_obs)

        self._update_meteor(pos_obs, pos_corr, mag_obs, mag_corr, scale=scale)
        self.valid_meteor = True
        self.draw()

    @abstractmethod
    def _update_meteor(self, pos_obs, pos_corr, mag_obs, mag_corr, *, scale=0.05):
        """ Inner method for updating the displayed meteor """

    def update_grid(self, x, y, grid, *, limit: float = 1, **kwargs):
        self._update_grid(x, y, grid, limit=limit, **kwargs)
        self.valid_grid = True
        self.draw()

    @abstractmethod
    def _update_grid(self, x, y, grid, *, limit: float = 1, **kwargs):
        """ Inner method for updating the grid """

    def clear_grid(self):
        empty = np.empty(shape=(0,))
        self.update_grid(empty, empty, np.empty(shape=(0, 2)))
