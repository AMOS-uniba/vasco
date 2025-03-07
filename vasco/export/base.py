import datetime
import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod

from matchers import Counsellor
from amosutils.projections import BorovickaProjection
from photometry import Calibration
from astropy.coordinates import EarthLocation


class Exporter(metaclass=ABCMeta):
    def __init__(self, matcher: Counsellor, location: EarthLocation, time: datetime.datetime,
                 projection: BorovickaProjection, calibration: Calibration):
        self._matcher: Counsellor = matcher
        self._location: EarthLocation = location
        self._time: datetime.datetime = time
        self._projection: BorovickaProjection = projection
        self._calibration: Calibration = calibration

    @abstractmethod
    def export(self, filename: str) -> None:
        pass

    @property
    def matcher(self) -> Counsellor:
        return self._matcher

    def _get_meteor(self):
        data = self.matcher.correct_meteor(self._projection, self._calibration)

        df = pd.DataFrame()
        df['ev_r'] = data.position_raw.alt.degree
        df['ev'] = data.position_corrected.alt.degree
        df['az_r'] = np.fmod(data.position_raw.az.degree, 360)
        df['az'] = np.fmod(data.position_corrected.az.degree, 360)
        df['fno'] = data.fnos
        df['b'] = 0
        df['bm'] = 0
        df['Lsum'] = 0
        df['mag_r'] = data.magnitudes_raw
        df['mag'] = data.magnitudes_corrected
        df['ra'] = 0
        df['dec'] = 0
        return df
