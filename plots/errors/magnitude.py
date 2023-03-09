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

    def norm(self, limit):
        return mpl.colors.TwoSlopeNorm(0, vmin=-2, vmax=2)

    def set_limits(self, errors):
        min_error = -Matcher.max_error(-errors)
        max_error = Matcher.max_error(errors)
        self.axis_alt.set_ylim([min_error * 1.05, max_error * 1.05])
        self.axis_az.set_ylim([min_error * 1.05, max_error * 1.05])
