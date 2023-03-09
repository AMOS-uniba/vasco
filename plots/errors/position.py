import numpy as np
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter

from plots.errors.base import BaseErrorPlot
from matchers import Matcher


class PositionErrorPlot(BaseErrorPlot):
    y_formatter = FuncFormatter(lambda x, pos: f'{x:.2f}°')
    cmap = mpl.cm.get_cmap('autumn_r')

    def add_axes(self):
        super().add_axes()
        self.axis_alt.set_ylim([0, None])
        self.axis_az.set_ylim([0, None])

    def norm(self, limit):
        return mpl.colors.Normalize(
            vmin=0,
            vmax=limit,
        )

    def set_limits(self, errors):
        max_error = Matcher.max_error(errors)
        self.axis_alt.set_ylim([0, max_error])
        self.axis_az.set_ylim([0, max_error])

    def update(self, positions, magnitudes, errors, *, limit=1):
        errors = np.degrees(errors)  # Convert errors in radians to degrees first
        super().update(positions, magnitudes, errors)
