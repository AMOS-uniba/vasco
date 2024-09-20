import matplotlib as mpl

from plots.sky.base import BaseSkyPlot


class MagnitudeSkyPlot(BaseSkyPlot):
    colour_stars = 'gray'
    cmap_stars = mpl.pyplot.get_cmap('coolwarm')
    cmap_meteors = mpl.pyplot.get_cmap('Blues_r')

    def norm(self, limit):
        return mpl.colors.TwoSlopeNorm(0, vmin=-2, vmax=2)
