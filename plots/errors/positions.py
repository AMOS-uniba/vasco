import matplotlib as mpl
from matplotlib.ticker import FuncFormatter

from plots.errors.base import BaseErrorPlot


class PositionErrorPlot(BaseErrorPlot):
    y_formatter = FuncFormatter(lambda x, pos: f'{x:.2f}°')
    cmap = mpl.cm.get_cmap('autumn_r')

    def add_axes(self):
        super().add_axes()
        self.axis_alt.set_ylim([0, None])
        self.axis_az.set_ylim([0, None])
