import copy
import numpy as np
import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord, AltAz

import colour as c
from logger import setupLog

log = setupLog(__name__)


class Catalogue:
    def __init__(self, stars=None, *, name=None):
        self.stars = pd.DataFrame()
        self.skycoord = None
        self.name = name

        if stars is None:
            self.stars = pd.DataFrame(dict(
                ra=pd.Series(dtype=float),
                dec=pd.Series(dtype=float),
                vmag=pd.Series(dtype=float),
            ))
        else:
            assert isinstance(stars, pd.DataFrame)
            self.stars = copy.deepcopy(stars)
        self.update_coord()
        self.reset_mask()

    def load(self, filename):
        self.name = filename
        self.stars = pd.read_csv(filename, sep='\t', header=1)
        self.update_coord()
        self.reset_mask()

    def update_coord(self):
        self.skycoord = SkyCoord(self.stars.ra.to_numpy() * u.deg, self.stars.dec.to_numpy() * u.deg)

    def filter_by_vmag(self, vmag):
        self.stars[self.stars.vmag <= vmag]['use'] = False
        return self

    def set_mask(self, condition):
        self.stars.loc[condition, 'use'] = False

    def reset_mask(self):
        if 'use' in self.stars.columns:
            self.stars.loc[:]['use'] = True
        else:
            self.stars['use'] = True
        log.info(f"Catalogue mask reset: {c.num(self.count_valid)} / {c.num(self.count)} stars used")

    def cull(self):
        """ Retain only currently unmasked data """
        log.info("Culling the catalogue")
        self.stars = self.stars[self.stars.use]
        self.update_coord()

    @property
    def count(self):
        return len(self.stars)

    @property
    def count_valid(self):
        return np.count_nonzero(self.stars.use)

    @property
    def mask(self):
        return self.stars.use

    @property
    def valid(self):
        return self.stars[self.mask]

    def altaz(self, location, time, *, masked: bool):
        altaz = AltAz(location=location, obstime=time, pressure=0, obswl=550 * u.nm)
        source = self.skycoord[self.mask] if masked else self.skycoord
        return source.transform_to(altaz)

    def to_altaz(self, location, time, *, masked: bool = True):
        """ Returns a packed (N, 2) np.ndarray with altitude and azimuth in radians """
        stars = self.altaz(location, time, masked=masked)
        return np.stack((stars.alt.radian, stars.az.radian), axis=1)

    def to_altaz_deg(self, location, time, *, masked: bool = True):
        """ Same but returns azimuth and altitude in degrees """
        stars = self.altaz(location, time, masked=masked)
        return np.stack((stars.alt.degree, stars.az.degree), axis=1)

    def to_altaz_chart(self, location, time, *, masked: bool = True):
        """ Same but returns azimuth in radians (t axis) and co-altitude in degrees (r axis) for chart display """
        stars = self.altaz(location, time, masked=masked)
        return np.stack((stars.az.radian, 90 - stars.alt.degree), axis=1)

    def vmag(self, *, masked: bool = True):
        return self.stars[self.mask].vmag.to_numpy() if masked else self.stars.vmag.to_numpy()

    def __str__(self):
        return f'<Catalogue "{c.name(self.name)}" with {c.num(self.count_valid)} of {c.num(self.count)} stars>'
