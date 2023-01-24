import numpy as np
import matplotlib as mpl

from matplotlib.ticker import MultipleLocator

from .base import BasePlot
from matchers import Matcher
from utilities import altaz_to_disk


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

        self.scatterCat = self.axis.scatter([0], [0], s=[1], marker='o', c='white')
#        self.scatterObs = self.axis.scatter([0], [0], s=[1], marker='x', c='red')
        self.quiver = None
        self.valid = False

    def update(self, cat, obs, *, limit=1, scale=0.05):
        cat = altaz_to_disk(cat)
        z, a = obs.T
        x = z * np.sin(a) / np.pi * 2
        y = -z * np.cos(a) / np.pi * 2
        obs = np.stack((x, y), axis=1)

        self.scatterCat.set_offsets(cat)
        self.scatterCat.set_sizes(np.ones_like(cat[:, 0]) * 4)
#        self.scatterObs.set_offsets(obs)
#        self.scatterObs.set_sizes(np.ones_like(obs[:, 0]) * 10)

        if self.quiver is not None:
            self.quiver.remove()

        norm = mpl.colors.Normalize(vmin=0, vmax=limit)
        self.quiver = self.axis.quiver(
            cat[:, 0], cat[:, 1],
            obs[:, 0] - cat[:, 0], obs[:, 1] - cat[:, 1],
            color=self.cmap(norm(np.sqrt((obs[:, 0] - cat[:, 0])**2 + (obs[:, 1] - cat[:, 1])**2))),
            scale=scale,
        )

        self.draw()

