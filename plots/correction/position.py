import numpy as np
import matplotlib as mpl

from .base import BaseCorrectionPlot


class PositionCorrectionPlot(BaseCorrectionPlot):
    def __init__(self, widget, **kwargs):
        super().__init__(widget, **kwargs)
        self.quiver_dots = None
        self.quiver_grid = None
        self.quiver_meteor = None

    def _update_dots(self, cat, obs, *, limit, scale):
        self.scatter_dots.set_offsets(cat)
        self.scatter_dots.set_sizes(np.ones_like(cat[:, 0]) * 4)

        if self.quiver_dots is not None:
            self.quiver_dots.remove()

        norm = mpl.colors.Normalize(vmin=0, vmax=limit)
        self.quiver_dots = self.axis.quiver(
            obs[:, 0], obs[:, 1],
            cat[:, 0] - obs[:, 0], cat[:, 1] - obs[:, 1],
            norm(np.sqrt((cat[:, 0] - obs[:, 0])**2 + (cat[:, 1] - obs[:, 1])**2)),
            cmap=self.cmap_dots,
            scale=scale,
        )

    def _update_meteor(self, obs, corr, magnitudes, scale=0.05):
        self.scatter_meteor.set_offsets(obs)
        self.scatter_meteor.set_sizes(np.ones_like(corr[:, 0]))

        if self.quiver_meteor is not None:
            self.quiver_meteor.remove()

        self.quiver_meteor = self.axis.quiver(
            obs[:, 0], obs[:, 1],
            corr[:, 0], corr[:, 1],
            magnitudes,
            cmap=self.cmap_meteor,
            scale=scale,
            width=0.002,
        )

    def _update_grid(self, x, y, grid, *, limit: float = 1):
        if self.quiver_grid is not None:
            self.quiver_grid.remove()

        u, v = grid[..., 0].ravel(), grid[..., 1].ravel()
        norm = mpl.colors.Normalize(vmin=0, vmax=limit)
        self.quiver_grid = self.axis.quiver(
            x, y, u, v, np.sqrt(u**2 + v**2),
            cmap=self.cmap_grid,
            width=0.0014,
        )

