import numpy as np

from .base import BasePlot


class SensorPlot(BasePlot):
    def add_axes(self):
        self.axis = self.figure.add_subplot()
        self.axis.set_xlim([-1, 1])
        self.axis.set_ylim([-1, 1])
        self.axis.grid(color='white', alpha=0.3)
        self.axis.set_aspect('equal')

        self.scatter = self.axis.scatter([0], [0], s=[50], c='white', marker='o')
        self.valid = False

    def update(self, data):
        self.axis.set_xlim([data.rect.xmin, data.rect.xmax])
        self.axis.set_ylim([data.rect.ymax, data.rect.ymin])
        self.scatter.set_offsets(data.positions)
        self.scatter.set_sizes(np.sqrt(data.intensities))
        self.draw()
