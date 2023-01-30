import numpy as np
import matplotlib as mpl

from .base import BasePlot


class SkyPlot(BasePlot):
    @staticmethod
    def to_chart(positions):
        return np.stack(
            (positions[:, 1], np.degrees(positions[:, 0])),
            axis=1,
        )

    def __init__(self, widget, **kwargs):
        self.cmap_stars = mpl.cm.get_cmap('autumn_r')
        self.cmap_meteors = mpl.cm.get_cmap('Blues_r')
        super().__init__(widget, **kwargs)

    def add_axes(self):
        self.axis = self.figure.add_subplot(projection='polar')
        self.axis.set_xlim([0, 2 * np.pi])
        self.axis.set_ylim([0, 90])
        self.axis.set_rlabel_position(0)
        self.axis.set_rticks([15, 30, 45, 60, 75])
        self.axis.yaxis.set_major_formatter('{x}°')
        self.axis.grid(color='white', alpha=0.3)
        self.axis.set_theta_offset(3 * np.pi / 2)

        self.scatterStars = self.axis.scatter([], [], s=[], c='white', marker='o')
        self.scatterDots = self.axis.scatter([], [], s=[], c='red', marker='x')
        self.scatterMeteor = self.axis.scatter([], [], s=[], c='cyan', marker='o')
        self.invalidate_stars()
        self.invalidate_dots()
        self.invalidate_meteor()

    def invalidate_stars(self):
        self.valid_stars = False

    def invalidate_dots(self):
        self.valid_dots = False

    def invalidate_meteor(self):
        self.valid_meteor = False

    def update_stars(self, positions, magnitudes):
        sizes = 0.2 * np.exp(-0.666 * (magnitudes - 5))
        self.scatterStars.set_offsets(positions)
        self.scatterStars.set_sizes(sizes)

        self.invalidate_stars()
        self.draw()

    def update_dots(self, positions, magnitudes, errors, *, limit=1):
        self.scatterDots.set_offsets(SkyPlot.to_chart(positions))

        norm = mpl.colors.Normalize(vmin=0, vmax=limit)
        self.scatterDots.set_facecolors(self.cmap_stars(norm(errors)))
        self.scatterDots.set_sizes(0.03 * magnitudes)

        self.invalidate_dots()
        self.draw()

    def update_meteor(self, positions, magnitudes):
        self.scatterMeteor.set_offsets(SkyPlot.to_chart(positions))

        norm = mpl.colors.Normalize(vmin=0, vmax=None)
        self.scatterMeteor.set_facecolors(self.cmap_meteors(norm(magnitudes)))
        self.scatterMeteor.set_sizes(0.0005 * magnitudes)

        self.invalidate_meteor()
        self.draw()

    def draw(self):
        """ Overrides default draw (has more validity indicators than base class """
        self.canvas.draw()
