import matplotlib as mpl
from matplotlib.ticker import FuncFormatter

from plots.errors.base import BaseErrorPlot
from matchers import Matcher


class MagnitudeErrorPlot(BaseErrorPlot):
    y_formatter = FuncFormatter(lambda x, pos: f'{x:+.1f}m')
    cmap_dots = mpl.cm.get_cmap('bwr')

    target: str = "star magnitudes"

    def norm(self, limit):
        return mpl.colors.TwoSlopeNorm(0, vmin=-2, vmax=2)

    def set_limits(self, errors):
        min_error = -Matcher.max_error(-errors)
        max_error = Matcher.max_error(errors)
        self.axis_alt.set_ylim([min_error * 1.05, max_error * 1.05])
        self.axis_az.set_ylim([min_error * 1.05, max_error * 1.05])
