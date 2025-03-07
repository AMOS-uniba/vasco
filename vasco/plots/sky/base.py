import math
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from abc import abstractmethod

from plots.base import BasePlot


class BaseSkyPlot(BasePlot):
    cmap_stars = plt.get_cmap('autumn_r')
    cmap_meteors = plt.get_cmap('Blues_r')
    colour_stars = 'white'
    colour_dots = 'red'
    colour_meteor = 'cyan'

    @staticmethod
    def to_chart(positions):
        return np.stack(
            (positions[:, 1], np.degrees(positions[:, 0])),
            axis=1,
        )

    @abstractmethod
    def norm(self, limit):
        """ Determine the norm to colour the data """

    def __init__(self, widget, **kwargs):
        self.scatter_stars = None
        self.scatter_dots = None
        self.scatter_meteor = None
        self.valid_stars = False
        self.valid_dots = False
        self.valid_meteor = False
        super().__init__(widget, **kwargs)

    def add_axes(self):
        self.axis = self.figure.add_subplot(projection='polar')
        self.axis.set_xlim([0, math.tau])
        self.axis.set_ylim([0, 90])
        self.axis.set_rlabel_position(0)
        self.axis.set_rticks([15, 30, 45, 60, 75])
        self.axis.yaxis.set_major_formatter('{x}°')
        self.axis.grid(color='white', alpha=0.3)
        self.axis.set_theta_offset(0.75 * math.tau)

        self.scatter_stars = self.axis.scatter([], [], s=[], c=self.colour_stars, marker='o')
        self.scatter_dots = self.axis.scatter([], [], s=[], c=self.colour_dots, marker='x')
        self.scatter_meteor = self.axis.scatter([], [], s=[], c=self.colour_meteor, marker='o')

    def invalidate(self):
        self.invalidate_stars()
        self.invalidate_dots()
        self.invalidate_meteor()

    def invalidate_stars(self):
        self.valid_stars = False

    def invalidate_dots(self):
        self.valid_dots = False

    def invalidate_meteor(self):
        self.valid_meteor = False

    def update_stars(self,
                     positions: np.ndarray[float],
                     magnitudes: np.ndarray[float]):
        sizes = 0.2 * np.exp(-0.833 * (magnitudes - 5))
        self.scatter_stars.set_offsets(positions)
        self.scatter_stars.set_sizes(sizes)
        self.valid_stars = True
        self.draw()

    def update_dots(self, positions, magnitudes, errors, *, limit=1):
        self.scatter_dots.set_offsets(self.to_chart(positions))

        self.scatter_dots.set_facecolors(self.cmap_stars(self.norm(limit)(errors)))
        self.scatter_dots.set_sizes(0.03 * magnitudes)

        self.valid_dots = True
        self.draw()

    def update_meteor(self, positions, magnitudes):
        self.scatter_meteor.set_offsets(self.to_chart(positions))

        norm = mpl.colors.Normalize(vmin=0, vmax=None)
        normalized = norm(magnitudes) if magnitudes.size > 0 else []
        self.scatter_meteor.set_facecolors(self.cmap_meteors(normalized))
        self.scatter_meteor.set_sizes(0.0005 * magnitudes)

        self.valid_meteor = True
        self.draw()
