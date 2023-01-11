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
        self.reset_mask()

    def filter_by_vmag(self, vmag):
        self.stars[self.stars.vmag <= vmag]['use'] = False
        return self

    def reset_mask(self):
        self.stars['use'] = True
        print(f"Catalogue mask reset: {self.count_valid} / {self.count} stars used")

    @property
    def count(self):
        return len(self.stars)

    @property
    def count_valid(self):
        return np.count_nonzero(self.stars.use)

    @property
    def vmag(self):
        return self.stars.vmag

    @property
    def valid(self):
        return self.stars[self.stars['use']]

    def to_altaz(self, location, time, masked):
        altaz = AltAz(location=location, obstime=time, pressure=0, obswl=500 * u.nm)
        source = self.skycoord[self.stars['use']] if masked else self.skycoord
        return source.transform_to(altaz)

    def to_altaz_deg(self, location, time, masked):
        stars = self.to_altaz(location, time, masked)
        return np.stack((stars.alt.degree, stars.az.degree), axis=1)

    def to_altaz_chart(self, location, time, masked):
        """ Same but returns azimuth in radians (t axis) and altitude in degrees (r axis) for chart display """
        stars = self.to_altaz(location, time, masked)
        return np.stack((stars.az.radian, 90 - stars.alt.degree), axis=1)


    def __str__(self):
        return f'<Catalogue "{self.name}" with {self.count} stars>'
