import copy
import datetime
import logging
import dotmap
import yaml
import numpy as np

from typing import Optional
from astropy.coordinates import EarthLocation
import astropy.units as u

from amosutils.projections.shifters import ScalingShifter
from .dotcollection import DotCollection
from .rect import Rect

log = logging.getLogger('vasco')


class SensorData:
    """ A set of stars and meteor snapshots in xy format """

    def __init__(self,
                 stars: Optional[DotCollection] = None,
                 meteor: Optional[DotCollection] = None,
                 *,
                 location: Optional[EarthLocation] = None,
                 timestamp: Optional[datetime.datetime] = None,
                 name: str = "(unknown)",
                 station: Optional[str] = None,
                 bounds: Optional[Rect] = None,
                 fps: int = 1):
        self.rect = Rect(-1, 1, -1, 1) if bounds is None else bounds
        self.shifter = ScalingShifter(x0=800, y0=600, xs=0.0044, ys=0.0044)

        self._stars_raw = DotCollection() if stars is None else stars
        self._stars_scaled = DotCollection() if stars is None else stars
        self._meteor_raw = DotCollection() if meteor is None else meteor
        self._meteor_scaled = DotCollection() if meteor is None else meteor
        self.name = "(unknown)" if name is None else name,
        self.fps = fps
        self.station = "(unknown station)" if station is None else station

        self.location = location
        self.timestamp = datetime.datetime.now() if timestamp is None else timestamp

    @staticmethod
    def load_YAML(file):
        data = dotmap.DotMap(yaml.safe_load(open(file, 'r')), _dynamic=False)

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
        location = EarthLocation(data.Longitude * u.deg, data.Latitude * u.deg, data.Altitude * u.m)
        station = data.Name

        return SensorData(
            stars, meteor,
            name=f"M{timestamp.strftime('%Y%m%d_%H%M%S')}_{station}_",
            bounds=Rect(0, w, 0, h),
            location=location,
            timestamp=timestamp,
            station=station,
            fps=data.FPS
        )

    def set_shifter_scales(self, xs, ys):
        self.shifter.xs = xs
        self.shifter.ys = ys

        self.rescale_stars()
        self.rescale_meteor()
        log.debug(f"Set shifter scales to xs = {xs:.6f}, ys = {ys:.6f}")

    def rescale_stars(self):
        self._stars_scaled = DotCollection(
            np.stack(self.shifter(self._stars_raw.xs(masked=False), self._stars_raw.ys(masked=False)), axis=1),
            self._stars_raw.intensities(masked=False),
            mask=self._stars_raw.mask,
        )

    def rescale_meteor(self):
        self._meteor_scaled = DotCollection(
            np.stack(self.shifter(self._meteor_raw.xs(masked=False), self._meteor_raw.ys(masked=False)), axis=1),
            self._meteor_raw.intensities(masked=False),
            fnos=self._meteor_raw.fnos(masked=False),
            mask=self._meteor_raw.mask,
        )

    def _collection_to_disk(self, collection, masked):
        return np.stack(self.shifter(collection.xs(masked), collection.ys(masked)), axis=1)

    def stars_to_disk(self, masked):
        return self._collection_to_disk(self.stars, masked)

    def meteor_to_disk(self, masked):
        return self._collection_to_disk(self.meteor, masked)

    def reset_mask(self):
        self._stars_scaled.mask = None

    @property
    def stars_raw(self):
        """ Stars in raw (pixel) coordinates, as detected """
        return self._stars_raw

    @property
    def stars(self):
        """ Stars in scaled (mm) coordinates """
        return self._stars_scaled

    @property
    def meteor_raw(self):
        """ Meteor in raw (pixel) coordinates, as detected """
        return self._meteor_raw

    @property
    def meteor(self):
        """ Meteor in scaled (mm) coordinates """
        return self._meteor_scaled

    def __str__(self):
        return f"<Sensor data with {self.stars.count_visible} / {self.stars.count} " \
               f"reference objects and {self.meteor.count} meteor snapshots>"
