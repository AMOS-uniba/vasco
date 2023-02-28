import matplotlib as mpl

from plots.sky.base import BaseSkyPlot


class PositionSkyPlot(BaseSkyPlot):
    def __init__(self, widget, **kwargs):
        super().__init__(widget, **kwargs)
        self.cmap_stars = mpl.cm.get_cmap('autumn_r')
        self.cmap_meteors = mpl.cm.get_cmap('Blues_r')
