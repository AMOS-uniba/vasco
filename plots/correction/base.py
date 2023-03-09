import numpy as np
import matplotlib as mpl

from abc import abstractmethod

from plots.base import BasePlot
from utilities import altaz_to_disk, proj_to_disk


class BaseCorrectionPlot(BasePlot):
    cmap_dots = mpl.cm.get_cmap('autumn_r')
    cmap_grid = mpl.cm.get_cmap('Greens')
    cmap_meteor = mpl.cm.get_cmap('Blues')
    colour_dots = 'white'
    colour_meteor = 'cyan'

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

        self.scatter_dots = self.axis.scatter([], [], s=[], marker='o', c=self.colour_dots)
        self.scatter_meteor = self.axis.scatter([], [], s=[], marker='o', c=self.colour_meteor)

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

    def update_dots(self, cat, obs, *, limit=1, scale=0.05):
        cat = altaz_to_disk(cat)
        obs = proj_to_disk(obs)

        self._update_dots(cat, obs, limit=limit, scale=scale)
        self.valid_dots = True
        self.draw()

    @abstractmethod
    def _update_dots(self, cat, obs, *, limit, scale):
        """ Inner method for updating the dots """

    def clear_errors(self):
        self.update_dots(None, None)

    def update_meteor(self, obs, corr, magnitudes, scale=0.05):
        obs = proj_to_disk(obs)

        self._update_meteor(obs, corr, magnitudes, scale)
        self.valid_meteor = True
        self.draw()

    @abstractmethod
    def _update_meteor(self, obs, corr, magnitudes, scale=0.05):
        """ Inner method for updating the displayed meteor """

    def update_grid(self, x, y, grid, *, limit: float = 1):
        self._update_grid(x, y, grid, limit=limit)
        self.valid_grid = True
        self.draw()

    @abstractmethod
    def _update_grid(self, x, y, grid, *, limit: float = 1):
        """ Inner method for updating the grid """

    def clear_grid(self):
        empty = np.empty(shape=(0,))
        self.update_grid(empty, empty, np.empty(shape=(0, 2)))
