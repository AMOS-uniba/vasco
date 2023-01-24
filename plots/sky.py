import numpy as np
import matplotlib as mpl

from .base import BasePlot


class SkyPlot(BasePlot):
    def __init__(self, widget, **kwargs):
        self.cmap = mpl.cm.get_cmap('autumn_r')
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

        self.starScatter = self.axis.scatter([0], [0], s=[50], c='white', marker='o')
        self.dotScatter = self.axis.scatter([0], [0], s=[50], c='red', marker='x')
        self.stars_valid = False
        self.dots_valid = False

    def update_dots(self, positions, magnitudes, errors, *, limit=1):
        z, a = positions.T
        self.dotScatter.set_offsets(np.stack((a, np.degrees(z)), axis=1))

        norm = mpl.colors.Normalize(vmin=0, vmax=limit)
        self.dotScatter.set_facecolors(self.cmap(norm(errors)))
        self.dotScatter.set_sizes(10 + 0.05 * magnitudes)

        self.dots_valid = True
        self.draw()

    def update_stars(self, positions, magnitudes):
        #loc = self.location.to_geodetic()
        #print(f"Plotting catalogue stars for {loc.lat:.6f}, {loc.lon:.6f} at {self.time}")
        sizes = 0.2 * np.exp(-0.666 * (magnitudes - 5))
        self.starScatter.set_offsets(positions)
        self.starScatter.set_sizes(sizes)

        self.stars_valid = True
        self.draw()

    def draw(self):
        """ Overrides default draw (has more validity indicators than base class """
        self.canvas.draw()
