import numpy as np
import matplotlib as mpl

from .base import BaseCorrectionPlot


class PositionCorrectionPlot(BaseCorrectionPlot):
    target = "star positions"

    def __init__(self, widget, **kwargs):
        super().__init__(widget, **kwargs)
        self.quiver_dots = None
        self.quiver_grid = None
        self.quiver_meteor = None

    def _update_dots(self, pos_obs, pos_cat, mag_obs, mag_cat, *, limit, scale):
        self.scatter_dots.set_offsets(pos_cat)
        self.scatter_dots.set_sizes(np.ones_like(pos_cat[:, 0]) * 4)

        if self.quiver_dots is not None:
            self.quiver_dots.remove()

        norm = mpl.colors.Normalize(vmin=0, vmax=limit)
        self.quiver_dots = self.axis.quiver(
            pos_obs[:, 0], pos_obs[:, 1],
            pos_cat[:, 0] - pos_obs[:, 0], pos_cat[:, 1] - pos_obs[:, 1],
            norm(np.sqrt((pos_cat[:, 0] - pos_obs[:, 0]) ** 2 + (pos_cat[:, 1] - pos_obs[:, 1]) ** 2)),
            cmap=self.cmap_dots,
            scale=scale / 1000,
        )

    def _update_meteor(self, pos_obs, pos_cat, mag_obs, mag_cat, scale=0.05):
        self.scatter_meteor.set_offsets(pos_obs)
        self.scatter_meteor.set_sizes(np.ones_like(pos_cat[:, 0]))

        if self.quiver_meteor is not None:
            self.quiver_meteor.remove()

        self.quiver_meteor = self.axis.quiver(
            pos_obs[:, 0], pos_obs[:, 1],
            pos_cat[:, 0], pos_cat[:, 1],
            mag_obs,
            cmap=self.cmap_meteor,
            scale=scale * 30,
            width=0.002,
        )

    def _update_grid(self, x, y, grid, *, limit: float = 1, **kwargs):
        if self.quiver_grid is not None:
            self.quiver_grid.remove()

        u, v = grid[..., 0].ravel(), grid[..., 1].ravel()
        self.quiver_grid = self.axis.quiver(
            x, y, u, v, np.sqrt(u**2 + v**2),
            cmap=self.cmap_grid,
            width=0.0014,
        )
