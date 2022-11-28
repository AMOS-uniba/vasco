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
        'sp': Station(-22.95341,  -68.17934, 2403),
        'pc': Station(-22.33535,  -68.64417, 2535),
        'hk': Station( 20.70740,  156.25615, 3068),
        'mk': Station( 19.82366,  155.47717, 4126),
    }
})
