import matplotlib as mpl

from plots.sky.base import BaseSkyPlot


class MagnitudeSkyPlot(BaseSkyPlot):
    def __init__(self, widget, **kwargs):
        super().__init__(widget, **kwargs)
        self.cmap_stars = mpl.cm.get_cmap('autumn_r')
        self.cmap_meteors = mpl.cm.get_cmap('Blues_r')
        self.cmap_veil = mpl.cm.get_cmap('coolwarm')
        self.veil = None

    def update_background(self, values):
        self.veil = self.axis.imshow(values, cmap=self.cmap_veil)