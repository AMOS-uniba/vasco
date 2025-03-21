import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from .base import BasePlot


class SensorPlot(BasePlot):
    cmap_meteors = plt.get_cmap('Blues_r')

    def __init__(self, widget, **kwargs):
        self.scatter_meteor = None
        self.scatter_stars = None
        self.valid = False
        super().__init__(widget, **kwargs)
        self.figure.tight_layout(rect=(0.05, 0, 1, 1))

    def add_axes(self):
        self.axis = self.figure.add_subplot()
        self.axis.set_xlim([-1, 1])
        self.axis.set_ylim([-1, 1])
        self.axis.grid(which='major', color='white', alpha=0.2)
        self.axis.grid(which='minor', color='white', alpha=0.12)
        self.axis.xaxis.set_major_locator(MultipleLocator(320))
        self.axis.xaxis.set_minor_locator(MultipleLocator(80))
        self.axis.yaxis.set_major_locator(MultipleLocator(240))
        self.axis.yaxis.set_minor_locator(MultipleLocator(80))
        self.axis.set_aspect('equal')

        self.scatter_stars = self.axis.scatter([], [], s=[], c='white', marker='o')
        self.scatter_meteor = self.axis.scatter([], [], s=[], c='cyan', marker='o')

    def update(self, data):
        self.axis.set_xlim([data.rect.xmin, data.rect.xmax])
        self.axis.set_ylim([data.rect.ymin, data.rect.ymax])
        self.scatter_stars.set_offsets(data.stars_raw.xy)
        self.scatter_stars.set_sizes(data.stars_raw.i / 100)
        self.scatter_meteor.set_offsets(data.meteor_raw.xy)

        norm = mpl.colors.Normalize(vmin=0, vmax=None)
        normalized = norm(data.meteor.i) if data.meteor.i.size > 0 else []
        self.scatter_meteor.set_facecolors(self.cmap_meteors(normalized))
        self.scatter_meteor.set_sizes(data.meteor.i / 2000)
        self.valid = True
        self.draw()

    def invalidate(self):
        self.valid = False
