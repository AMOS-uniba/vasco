import numpy as np
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter

from plots.errors.base import BaseErrorPlot
from matchers import Matcher


class MagnitudeErrorPlot(BaseErrorPlot):
    y_formatter = FuncFormatter(lambda x, pos: f'{x:+.2f}m')
    cmap = mpl.cm.get_cmap('bwr')

    def __init__(self, widget, **kwargs):
        super().__init__(widget, **kwargs)

    def add_axes(self):
        super().add_axes()
        self.axis_alt.set_ylim([0, None])
        self.axis_az.set_ylim([0, None])

    def update(self, positions, magnitudes, errors, *, limit=1):
        alt = np.degrees(positions[:, 0])
        az = np.degrees(positions[:, 1])

        min_error = -Matcher.max_error(-errors)
        max_error = Matcher.max_error(errors)

        if max_error is not np.nan:
            self.axis_alt.set_ylim([min_error * 1.05, max_error * 1.05])
            self.axis_az.set_ylim([min_error * 1.05, max_error * 1.05])
            self.scatter_alt.set_offsets(np.stack((alt, errors), axis=1))
            self.scatter_az.set_offsets(np.stack((az, errors), axis=1))

            norm = mpl.colors.TwoSlopeNorm(0, vmin=-2, vmax=2)
            self.scatter_alt.set_facecolors(self.cmap(norm(errors)))
            self.scatter_alt.set_sizes(0.05 * magnitudes)
            self.scatter_az.set_facecolors(self.cmap(norm(errors)))
            self.scatter_az.set_sizes(0.05 * magnitudes)

        self.draw()
