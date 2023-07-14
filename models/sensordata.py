import copy
import datetime
import numpy as np

import colour as c

from typing import Optional

from .dotcollection import DotCollection
from projections.shifters import ScalingShifter
from .rect import Rect


class SensorData:
    """ A set of stars and meteor snapshots in xy format """

    def __init__(self,
                 stars: Optional[DotCollection] = None,
                 meteor: Optional[DotCollection] = None,
                 *,
                 id: str = "(unknown)",
                 station: Optional[str] = None,
                 timestamp: Optional[datetime.datetime] = None,
                 bounds: Optional[Rect] = None,
                 fps: int = 1):
        self.rect = Rect(-1, 1, -1, 1) if bounds is None else bounds
        self._stars = DotCollection() if stars is None else stars
        self._meteor = DotCollection() if meteor is None else meteor
        #self.stars_shifted = DotCollection(star_positions, star_intensities, None)
        #self.meteor_shifted = DotCollection(meteor_positions, meteor_intensities, None)
        self.id = "(unknown)" if id is None else id,
        self.fps = fps
        self.timestamp = datetime.datetime.now() if timestamp is None else timestamp
        self.station = "(unknown station)" if station is None else station
        self.shifter = ScalingShifter(x0=800, y0=600, xs=0.0044, ys=0.0044)

        # temporary fix
        self.set_shifter_scales(0.0044, 0.0044)

    @staticmethod
    def load(data):
#        assert np.min(np.asarray([star.intensity for star in data.Refstars])) > 0,\
#            "All reference stars must have positive intensity!"
#        assert np.min(np.asarray([snapshot.intensity for snapshot in data.Trail])) > 0,\
#            "All meteor snapshots must have positive intensity!"

        w, h = tuple(map(int, data.Resolution.split('x')))
        stars = DotCollection(
            np.asarray([[star.x, star.y] for star in data.Refstars]),
            np.asarray([star.intensity for star in data.Refstars]),
        )
        meteor = DotCollection(
            np.asarray([[snapshot.xc, snapshot.yc] for snapshot in data.Trail]),
            np.asarray([snapshot.intensity for snapshot in data.Trail]),
            fnos=np.asarray([snapshot.fno for snapshot in data.Trail], dtype=int)
        )
        timestamp = datetime.datetime.strptime(data.EventStartTime, "%Y-%m-%d %H:%M:%S.%f")
        station = data.Name

        return SensorData(
            stars, meteor,
            id=f"M{timestamp.strftime('%Y%m%d_%H%M%S')}_{station}_",
            bounds=Rect(0, w, 0, h),
            timestamp=timestamp,
            station=station,
            fps=data.FPS
        )

    def set_shifter_scales(self, xs, ys):
        self.shifter.xs = xs
        self.shifter.ys = ys
        self.stars_shifted = DotCollection(
            np.stack(self.shifter(self._stars.xs(masked=False), self._stars.ys(masked=False)), axis=1),
            self._stars.intensities(masked=False),
            mask=self._stars.mask,
        )
        self.meteor_shifted = DotCollection(
            np.stack(self.shifter(self._meteor.xs(masked=False), self._meteor.ys(masked=False)), axis=1),
            self._meteor.intensities(masked=False),
            fnos=self._meteor.fnos(False),
            mask=self._meteor.mask,
        )

    def _collection_to_disk(self, collection, masked):
        return np.stack(self.shifter(collection.xs(masked), collection.ys(masked)), axis=1)

    def stars_to_disk(self, masked):
        return self._collection_to_disk(self.stars, masked)

    def meteor_to_disk(self, masked):
        return self._collection_to_disk(self.meteor, masked)

    def reset_mask(self):
        self.stars.mask = None

    def culled_copy(self):
        out = copy.deepcopy(self)
        out.stars.cull()
        return out

    @property
    def stars(self):
        return self.stars_shifted

    @property
    def meteor(self):
        return self.meteor_shifted

    def __str__(self):
        return f"<Sensor data with {c.num(self.stars.count_valid)} / {c.num(self.stars.count)} " \
               f"reference stars and {c.num(self.meteor.count)} meteor snapshots>"
