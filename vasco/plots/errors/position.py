import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from plots.errors.base import BaseErrorPlot
from plots.base import cmap_gyr
from models import Matcher


class PositionErrorPlot(BaseErrorPlot):
    y_formatter = FuncFormatter(lambda x, pos: f'{x:.3f}°')
    cmap_dots = cmap_gyr

    target: str = "star positions"

    def add_axes(self):
        super().add_axes()
        self.hline_alt = self.axis_alt.axhline(0)
        self.hline_az = self.axis_az.axhline(0)

    def norm(self, limit):
        return mpl.colors.Normalize(vmin=0, vmax=limit)

    def set_limits(self, errors):
        max_error = Matcher.max_error(errors)
        if np.isnan(max_error) or not np.isfinite(max_error):
            max_error = 180 # On an empty set show the entire range
        else:
            max_error = np.minimum(180, max_error * 1.05) # Otherwise just add some small margin

        self.axis_alt.set_ylim([0, max_error])
        self.axis_az.set_ylim([0, max_error])

    def update_dots(self, positions, magnitudes, errors, *, limit=1):
        errors = np.degrees(errors)  # Convert errors in radians to degrees first

        self.hline_alt.remove()
        self.hline_alt = self.axis_alt.axhline(limit, c='red', lw=0.5)
        self.hline_az.remove()
        self.hline_az = self.axis_az.axhline(limit, c='red', lw=0.5)

        super().update_dots(positions, magnitudes, errors, limit=limit)
