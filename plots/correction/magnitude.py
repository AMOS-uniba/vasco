import numpy as np
import matplotlib as mpl

from .base import BaseCorrectionPlot


class MagnitudeCorrectionPlot(BaseCorrectionPlot):
    cmap_grid = mpl.cm.get_cmap('bwr')
    norm_grid = mpl.colors.TwoSlopeNorm(0, vmin=-2, vmax=2)

    def __init__(self, widget, **kwargs):
        super().__init__(widget, **kwargs)
        self.magnitude_dots = None
        self.magnitude_grid = None
        self.magnitude_meteor = None

    def _update_dots(self, pos_obs, pos_cat, mag_obs, mag_cat, *, limit, scale):
        self.scatter_dots.set_offsets(pos_cat)
        self.scatter_dots.set_sizes(np.ones_like(pos_cat[:, 0]) * 20)
        self.scatter_dots.set_facecolors(self.cmap_dots(self.norm_grid(mag_cat - mag_obs)))

    def _update_meteor(self, pos_obs, pos_corr, mag_obs, mag_corr, scale=0.05):
        self.scatter_meteor.set_offsets(pos_obs)
        self.scatter_meteor.set_sizes(np.exp(np.ones_like(mag_obs)))
        self.scatter_meteor.set_facecolors(self.cmap_meteor(self.norm_grid(mag_obs - mag_corr)))

    def _update_grid(self, x, y, grid, *, limit: float = 1):
        if self.magnitude_grid is not None:
            self.magnitude_grid.remove()

        self.magnitude_grid = self.axis.imshow(
            grid[..., 0],
            cmap=self.cmap_grid,
            norm=self.norm_grid,
            extent=[-1, 1, -1, 1],
            interpolation='bicubic',
        )
