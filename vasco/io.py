"""
Reading a sighting file.

This was `SensorData.load_YAML`, and it stayed behind when SensorData moved to demeteor. A reader
for a station's file format is a different thing from a container for what it holds, and there is a
reason not to move it yet: **there are two hand-written readers for this format in this ecosystem**
-- this one, and `Metadata._parse_kvant` in the AMOS server, which reads only the header and counts
the stars without storing them. They have already drifted apart once: this one read `Refstars`, the
server's read `Stars`, which no camera has ever written, so the server's star count was None for
every report it ever ingested.

One reader is the fix, and its home is demeteor now that SensorData lives there. That is a separate
piece of work because it has to decide what demeteor is for -- it would mean the library learning
to parse UFO's XML as well -- and a decision like that is better taken deliberately than as the
side effect of a move.
"""
import datetime
import logging

import astropy.units as u
import dotmap
import numpy as np
from astropy.coordinates import EarthLocation
from demeteor.sensor import DotCollection, Rect, SensorData

from vasco import yaml_io

log = logging.getLogger('vasco')


def load_sighting(file) -> SensorData:
    """
    One Kvant sighting file into a SensorData: the reference stars, the trail, and where and when.

    Read in YAML 1.2, never 1.1 -- see vasco/yaml_io.py. Every AMOS file zero-pads its frame
    numbers, and a leading zero is not an octal prefix.
    """
    data = dotmap.DotMap(yaml_io.load(file), _dynamic=False)

    w, h = tuple(map(int, data.Resolution.split('x')))
    stars = DotCollection(
        np.asarray([[star.x, star.y] for star in data.Refstars]),
        np.asarray([star.intensity for star in data.Refstars]),
    )
    meteor = DotCollection(
        np.asarray([[snapshot.xc, snapshot.yc] for snapshot in data.Trail]),
        np.asarray([snapshot.intensity for snapshot in data.Trail]),
        fnos=np.asarray([snapshot.fno for snapshot in data.Trail], dtype=int),
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
        fps=data.FPS,
    )
