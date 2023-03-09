import numpy as np
import matplotlib as mpl

from .base import BaseCorrectionPlot


class MagnitudeCorrectionPlot(BaseCorrectionPlot):
    cmap_grid = mpl.cm.get_cmap('bwr')

    def __init__(self, widget, **kwargs):
        super().__init__(widget, **kwargs)
        self.magnitude_dots = None
        self.magnitude_grid = None
        self.magnitude_meteor = None

    def _update_dots(self, cat, obs, *, limit, scale):
        self.scatter_dots.set_offsets(cat)
        self.scatter_dots.set_sizes(np.ones_like(cat[:, 0]) * 4)
        self.scatter_dots.set_facecolors('lime')

    def _update_meteor(self, obs, corr, magnitudes, scale=0.05):
        self.scatter_meteor.set_offsets(obs)
        self.scatter_meteor.set_sizes(np.ones_like(corr[:, 0]))
        norm = mpl.colors.TwoSlopeNorm(0)
        self.scatter_meteor.set_facecolors(self.cmap_meteor(norm(magnitudes)))

    def _update_grid(self, x, y, grid, *, limit: float = 1):
        if self.magnitude_grid is not None:
            self.magnitude_grid.remove()

        norm = mpl.colors.TwoSlopeNorm(0)
        self.magnitude_grid = self.axis.imshow(
            grid[..., 0],
            cmap=self.cmap_grid,
            norm=norm,
            extent=[-1, 1, -1, 1],
            interpolation='bicubic',
        )
