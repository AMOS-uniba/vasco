import numpy as np

from .base import BasePlot


class SensorPlot(BasePlot):
    def __init__(self, widget, **kwargs):
        self.scatter_meteor = None
        self.scatter_stars = None
        self.valid = False
        super().__init__(widget, **kwargs)

    def add_axes(self):
        self.axis = self.figure.add_subplot()
        self.axis.set_xlim([-1, 1])
        self.axis.set_ylim([-1, 1])
        self.axis.grid(color='white', alpha=0.2)
        self.axis.set_aspect('equal')

        self.scatter_stars = self.axis.scatter([], [], s=[], c='white', marker='o')
        self.scatter_meteor = self.axis.scatter([], [], s=[], c='cyan', marker='o')

    def update(self, data):
        self.axis.set_xlim([data.rect.xmin, data.rect.xmax])
        self.axis.set_ylim([data.rect.ymax, data.rect.ymin])
        self.scatter_stars.set_offsets(data.stars.xy)
        self.scatter_stars.set_sizes(data.stars.m / 100)
        self.scatter_meteor.set_offsets(data.meteor.xy)
        self.scatter_meteor.set_sizes(data.meteor.m / 10
        self.valid = True
        self.draw()

    def invalidate(self):
        self.valid = False


