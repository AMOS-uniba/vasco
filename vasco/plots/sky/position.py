import matplotlib as mpl

from vasco.plots.sky.base import BaseSkyPlot
from vasco.plots.base import cmap_gyr


class PositionSkyPlot(BaseSkyPlot):
    cmap_stars = cmap_gyr
    cmap_meteors = mpl.pyplot.get_cmap('Blues_r')

    def norm(self, limit):
        return mpl.colors.Normalize(vmin=0, vmax=limit)
