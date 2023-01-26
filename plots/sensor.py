import numpy as np

from .base import BasePlot


class SensorPlot(BasePlot):
    def add_axes(self):
        self.axis = self.figure.add_subplot()
        self.axis.set_xlim([-1, 1])
        self.axis.set_ylim([-1, 1])
        self.axis.grid(color='white', alpha=0.2)
        self.axis.set_aspect('equal')

        self.scatterStars = self.axis.scatter([], [], s=[], c='white', marker='o')
        self.scatterMeteor = self.axis.scatter([], [], s=[], c='cyan', marker='o')
        self.valid = False

    def update(self, data):
        self.axis.set_xlim([data.rect.xmin, data.rect.xmax])
        self.axis.set_ylim([data.rect.ymax, data.rect.ymin])
        self.scatterStars.set_offsets(data.stars.xy)
        self.scatterStars.set_sizes(np.sqrt(data.stars.m / 100))
        self.scatterMeteor.set_offsets(data.meteor.xy)
        self.scatterMeteor.set_sizes(np.sqrt(data.meteor.m / 100))
        self.draw()

