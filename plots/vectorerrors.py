import numpy as np
import matplotlib as mpl

from matplotlib.ticker import MultipleLocator

from .base import BasePlot
from matchers import Matcher
from utilities import altaz_to_disk, proj_to_disk, by_azimuth


class VectorErrorPlot(BasePlot):
    def __init__(self, widget, **kwargs):
        self.cmap = mpl.cm.get_cmap('autumn_r')
        super().__init__(widget, **kwargs)

    def add_axes(self):
        self.axis = self.figure.add_subplot()
        self.axis.set_xlim([-1, 1])
        self.axis.set_ylim([-1, 1])
        self.axis.set_xlabel('x')
        self.axis.set_ylabel('y')
        self.axis.set_aspect('equal')
        self.axis.grid(color='white', alpha=0.2)

        self.scatterCat = self.axis.scatter([], [], s=[], marker='o', c='white')
        self.quiver_dots = None
        self.quiver_grid = None
        self.valid_dots = False
        self.valid_grid = False

    def update_dots(self, cat, obs, *, limit=1, scale=0.05):
        cat = altaz_to_disk(cat)
        obs = proj_to_disk(obs)

        self.scatterCat.set_offsets(cat)
        self.scatterCat.set_sizes(np.ones_like(cat[:, 0]) * 4)

        if self.quiver_dots is not None:
            self.quiver_dots.remove()

        norm = mpl.colors.Normalize(vmin=0, vmax=limit)
        self.quiver_dots = self.axis.quiver(
            cat[:, 0], cat[:, 1],
            obs[:, 0] - cat[:, 0], obs[:, 1] - cat[:, 1],
            color=self.cmap(norm(np.sqrt((obs[:, 0] - cat[:, 0])**2 + (obs[:, 1] - cat[:, 1])**2))),
            scale=scale,
        )
        self.draw()

    def update_grid(self, x, y, u, v):
        if self.quiver_grid is not None:
            self.quiver_grid.remove()

        self.quiver_grid = self.axis.quiver(
            x, y, u, v, color=by_azimuth(np.stack((u, v), axis=1)), width=0.002
        )

        self.draw()

