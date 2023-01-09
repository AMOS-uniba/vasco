import numpy as np
import pandas as pd

from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, AltAz


class Catalogue():
    def __init__(self, filename=None):
        self.stars = None
        self.skycoords = None
        self.name = None

        if filename is not None:
            self.load(filename)

    def load(self, filename):
        self.name = filename
        self.stars = pd.read_csv(filename, sep='\t', header=1)
        self.skycoord = SkyCoord(self.stars.ra * u.deg, self.stars.dec * u.deg)
        self.stars['use'] = True

    def filter(self, vmag):
        self.stars[self.stars.vmag <= vmag]['use'] = False
        return self

    @property
    def valid_stars(self):
        return self.stars[self.stars['use']]

    def to_altaz(self, location, time):
        altaz = AltAz(location=location, obstime=time, pressure=0, obswl=500 * u.nm)
        stars = self.skycoord[self.stars['use']].transform_to(altaz)
        return np.stack((stars.alt.degree, stars.az.degree))

    @property
    def vmag(self):
        return self.stars.vmag

    @property
    def count(self):
        return len(self.stars)

    @property
    def valid(self):
        return self.stars[self.stars['use']]

    def __str__(self):
        return f'<Catalogue "{self.name}" with {self.count} stars>'
