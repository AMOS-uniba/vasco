import matplotlib as mpl

from plots.errors.base import BaseErrorPlot


class MagnitudeErrorPlot(BaseErrorPlot):
    def __init__(self, widget, **kwargs):
        self.cmap = mpl.cm.get_cmap('coolwarm')
        super().__init__(widget, **kwargs)
