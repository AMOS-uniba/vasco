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
        self.set_stars(pd.read_csv(filename, sep='\t', header=1))

    def set_stars(self, stars):
        self.stars = stars
        self.skycoords = SkyCoord(self.stars.ra * u.deg, self.stars.dec * u.deg)

    def filter(self, vmag):
        self.set_stars(self.stars[self.stars.vmag <= vmag])
        return self

    def to_altaz(self, location, time):
        altaz = AltAz(location=location, obstime=time, pressure=101325 * u.pascal, obswl=500 * u.nm)
        stars = self.skycoords.transform_to(altaz)
        return np.stack((stars.alt.degree, stars.az.degree))

    @property
    def vmag(self):
        return self.stars.vmag

    @property
    def count(self):
        return len(self.stars)

    def __str__(self):
        return f"<Catalogue \"{self.name}\" with {self.count} stars>"
