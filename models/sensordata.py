import copy
import numpy as np

from models.dotcollection import DotCollection
from projections.shifters import ScalingShifter


class Rect:
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.xcen = (xmax + xmin) / 2
        self.ycen = (ymax + ymin) / 2
        self.r = np.sqrt((ymax - ymin)**2 + (xmax - xmin)**2) / 2
        self.shifter = ScalingShifter(x0=self.xcen, y0=self.ycen, scale=1 / self.r)

    def to_unit(self, data):
        """ data: np.ndarray(N, 2) """
        out = np.zeros_like(data)
        out[:, 0] = (data[:, 0] - self.xcen) / self.r
        out[:, 1] = (data[:, 1] - self.ycen) / self.r
        return out


class SensorData:
    """ A set of stars and optionally a meteor in xy format """

    def __init__(self, star_positions=None, star_intensities=None, meteor_positions=None, meteor_intensities=None):
        self.rect = Rect(-1, 1, -1, 1)
        self.stars = DotCollection(star_positions, star_intensities, None)
        self.meteor = DotCollection(meteor_positions, meteor_intensities, None)

    def load(self, data):
        w, h = tuple(map(int, data.Resolution.split('x')))
        self.rect = Rect(0, w, 0, h)
        self.stars = DotCollection(
            np.asarray([[star.x, star.y] for star in data.Refstars]),
            np.asarray([star.intensity for star in data.Refstars]),
        )
        self.meteor = DotCollection(
            np.asarray([[snapshot.xc, snapshot.yc] for snapshot in data.Trail]),
            np.asarray([snapshot.intensity for snapshot in data.Trail]),
        )

    def collection_to_disk(self, collection, masked):
        return np.stack(self.rect.shifter(collection.xs(masked), collection.ys(masked)), axis=1)

    def stars_to_disk(self, masked):
        return self.collection_to_disk(self.stars, masked)

    def meteor_to_disk(self, masked):
        return self.collection_to_disk(self.meteor, masked)

    def reset_mask(self):
        self.stars.mask = None

    def culled_copy(self):
        out = copy.deepcopy(self)
        out.stars.cull()
        return out

    def __str__(self):
        return f"<Sensor data with {self.stars.count_valid} / {self.stars.count} " \
               f"reference stars and {self.meteor.count} meteor snapshots>"
