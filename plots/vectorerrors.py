import numpy as np
import matplotlib as mpl


from astropy.coordinates import AltAz
from .base import BasePlot
from utilities import altaz_to_disk, proj_to_disk


class VectorErrorPlot(BasePlot):
    def __init__(self, widget, **kwargs):
        self.cmap_dots = mpl.cm.get_cmap('autumn_r')
        self.cmap_grid = mpl.cm.get_cmap('Greens')
        self.cmap_meteor = mpl.cm.get_cmap('Blues')
        super().__init__(widget, **kwargs)

    def add_axes(self):
        self.axis = self.figure.add_subplot()
        self.axis.set_xlim([-1, 1])
        self.axis.set_ylim([-1, 1])
        self.axis.set_xlabel('x')
        self.axis.set_ylabel('y')
        self.axis.set_aspect('equal')
        self.axis.grid(color='white', alpha=0.2)

        self.scatterDots = self.axis.scatter([], [], s=[], marker='o', c='white')
        self.scatterMeteor = self.axis.scatter([], [], s=[], marker='o', c='cyan')
        self.quiver_dots = None
        self.quiver_grid = None
        self.quiver_meteor = None

    def invalidate_dots(self):
        self.valid_dots = False

    def invalidate_grid(self):
        self.valid_grid = False

    def invalidate_meteor(self):
        self.valid_meteor = False

    def update_errors(self, cat, obs, *, limit=1, scale=0.05):
        cat = altaz_to_disk(cat)
        obs = proj_to_disk(obs)

        self.scatterDots.set_offsets(cat)
        self.scatterDots.set_sizes(np.ones_like(cat[:, 0]) * 4)

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

        self.valid_dots = True
        self.draw()

    def clear_errors(self):
        self.update_errors(None, None)

    def update_meteor(self, obs, corr, magnitudes, scale=0.05):
        obs = proj_to_disk(obs)

        self.scatterMeteor.set_offsets(obs)
        self.scatterMeteor.set_sizes(np.ones_like(corr[:, 0]))

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

        self.valid_meteor = True
        self.draw()

    def update_grid(self, x, y, u, v, *, limit=1):
        if self.quiver_grid is not None:
            self.quiver_grid.remove()

        norm = mpl.colors.Normalize(vmin=0, vmax=limit)
        self.quiver_grid = self.axis.quiver(
            x, y, u, v, np.sqrt(u**2 + v**2),
            cmap=self.cmap_grid,
            #color=by_azimuth(np.stack((u, v), axis=1)),
            width=0.0014,
        )

        self.valid_grid = True
        self.draw()

    def clear_grid(self):
        self.update_grid(
            np.empty(shape=(0,)),
            np.empty(shape=(0,)),
            np.empty(shape=(0,)),
            np.empty(shape=(0,)),
        )


