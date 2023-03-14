import numpy as np
import matplotlib as mpl

from .base import BaseCorrectionPlot


class MagnitudeCorrectionPlot(BaseCorrectionPlot):
    cmap_dots = mpl.cm.get_cmap('bwr')
    cmap_grid = mpl.cm.get_cmap('bwr')
    norm_grid = mpl.colors.TwoSlopeNorm(0, vmin=-2, vmax=2)
    target = "star magnitudes"

    def __init__(self, widget, **kwargs):
        super().__init__(widget, **kwargs)
        self.magnitude_dots = None
        self.magnitude_grid = None
        self.magnitude_meteor = None

    def _update_scatter(self, scatter, pos_obs, mag_obs, dmags, *, cmap):
        scatter.set_offsets(pos_obs)
        scatter.set_sizes(np.exp(-mag_obs / 3) * 100)
        scatter.set_facecolors(cmap(self.norm_grid(dmags)))

    def _update_dots(self, pos_obs, pos_cat, mag_obs, mag_cat, *, limit, scale):
        self._update_scatter(self.scatter_dots, pos_obs, mag_obs, mag_cat - mag_obs, cmap=self.cmap_dots)

    def _update_meteor(self, pos_obs, pos_corr, mag_obs, mag_corr, *, scale=0.05):
        self._update_scatter(self.scatter_meteor, pos_obs, mag_obs, mag_obs - mag_corr, cmap=self.cmap_meteor)

    def _update_grid(self, x, y, grid, *, limit: float = 1, **kwargs):
        if self.magnitude_grid is not None:
            self.magnitude_grid.remove()

        xres, yres, zres = grid.shape
        assert xres == yres, f"Magnitude grid shape is not square, is ({xres}, {yres}, {zres})"
        assert zres == 1, f"Magnitude grid shape should be (R, R, 1), is ({xres}, {yres}, {zres})"

        xext = (xres + 1) / xres
        yext = (yres + 1) / yres
        self.magnitude_grid = self.axis.imshow(
            grid[..., 0],
            cmap=self.cmap_grid,
            norm=self.norm_grid,
            extent=[-xext, xext, -yext, yext],
            origin='lower',
            interpolation=kwargs.get('interpolation', 'nearest'),
        )
