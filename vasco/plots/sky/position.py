import matplotlib as mpl

from plots.sky.base import BaseSkyPlot
from plots.base import cmap_gyr


class PositionSkyPlot(BaseSkyPlot):
    cmap_stars = cmap_gyr#mpl.pyplot.get_cmap('autumn_r')
    cmap_meteors = mpl.pyplot.get_cmap('Blues_r')

    def norm(self, limit):
        return mpl.colors.Normalize(vmin=0, vmax=limit)
