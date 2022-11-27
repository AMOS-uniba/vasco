import dotmap
import astropy
from astropy.coordinates import EarthLocation
from astropy import units as u


class Station:
    def __init__(self, latitude, longitude, altitude):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude

    def earth_location(self):
        return EarthLocation(self.longitude * u.deg, self.latitude * u.deg, self.altitude * u.m)


AMOS = dotmap.DotMap({
    'stations': {
        'sp': Station(-22.95341, -68.17934, 2403),
        'pc': Station(-22.7, -68.5, 2535),
    }
})
