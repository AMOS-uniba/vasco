import matplotlib as mpl

from plots.sky.base import BaseSkyPlot


class MagnitudeSkyPlot(BaseSkyPlot):
    colour_stars = 'gray'

    def __init__(self, widget, **kwargs):
        super().__init__(widget, **kwargs)
        self.cmap_stars = mpl.cm.get_cmap('coolwarm')
        self.cmap_meteors = mpl.cm.get_cmap('Blues_r')
        self.cmap_veil = mpl.cm.get_cmap('coolwarm')
        self.veil = None

    def update_background(self, values):
        self.veil = self.axis.imshow(values, cmap=self.cmap_veil)

    def norm(self, limit):
        return mpl.colors.TwoSlopeNorm(0, vmin=-2, vmax=2)
