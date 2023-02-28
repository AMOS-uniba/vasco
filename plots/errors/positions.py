import matplotlib as mpl

from plots.errors.base import BaseErrorPlot


class PositionErrorPlot(BaseErrorPlot):
    def __init__(self, widget, **kwargs):
        super().__init__(widget, **kwargs)
        self.cmap = mpl.cm.get_cmap('autumn_r')
