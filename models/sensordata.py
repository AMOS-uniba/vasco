import copy
import datetime
import numpy as np

import colour as c
from .dotcollection import DotCollection
from .rect import Rect


class SensorData:
    """ A set of stars and meteor snapshots in xy format """

    def __init__(self, star_positions=None, star_intensities=None, meteor_positions=None, meteor_intensities=None):
        self.rect = Rect(-1, 1, -1, 1)
        self.stars = DotCollection(star_positions, star_intensities, None)
        self.meteor = DotCollection(meteor_positions, meteor_intensities, None)

    def load(self, data):
        assert np.min(np.asarray([star.intensity for star in data.Refstars])) > 0,\
            "All reference stars must have positive intensity!"
        assert np.min(np.asarray([snapshot.intensity for snapshot in data.Trail])) > 0,\
            "All meteor snapshots must have positive intensity!"

        w, h = tuple(map(int, data.Resolution.split('x')))
        self.rect = Rect(0, w, 0, h)
        self.time = datetime.datetime.strptime(data.EventStartTime, "%Y-%m-%d %H:%M:%S.%f")
        self.stars = DotCollection(
            np.asarray([[star.x, star.y] for star in data.Refstars]),
            np.asarray([star.intensity for star in data.Refstars]),
        )
        self.meteor = DotCollection(
            np.asarray([[snapshot.xc, snapshot.yc] for snapshot in data.Trail]),
            np.asarray([snapshot.intensity for snapshot in data.Trail]),
            fnos=np.asarray([snapshot.fno for snapshot in data.Trail], dtype=int)
        )
        self.station = data.Name
        self.id = f"M{self.time.strftime('%Y%m%d_%H%M%S')}_{self.station}_"

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
        return f"<Sensor data with {c.num(self.stars.count_valid)} / {c.num(self.stars.count)} " \
               f"reference stars and {c.num(self.meteor.count)} meteor snapshots>"
