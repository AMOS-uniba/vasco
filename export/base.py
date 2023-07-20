import datetime
import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod

from matchers import Counselor
from projections import BorovickaProjection
from photometry import Calibration
from astropy.coordinates import EarthLocation


class Exporter(metaclass=ABCMeta):
    def __init__(self, matcher: Counselor, location: EarthLocation, time: datetime.datetime,
                 projection: BorovickaProjection, calibration: Calibration):
        self._matcher: Counselor = matcher
        self._location: EarthLocation = location
        self._time: datetime.datetime = time
        self._projection: BorovickaProjection = projection
        self._calibration: Calibration = calibration

    @abstractmethod
    def export(self, filename: str) -> None:
        pass

    def _get_meteor(self):
        data = self._matcher.correct_meteor(self._projection, self._calibration)

        df = pd.DataFrame()
        df['ev_r'] = 90 - data.position_raw.alt.degree
        df['ev'] = 90 - data.position_corrected.alt.degree
        df['az_r'] = np.fmod(data.position_raw.az.degree + 180, 360)
        df['az'] = np.fmod(data.position_corrected.az.degree + 180, 360)
        df['fno'] = data.fnos
        df['b'] = 0
        df['bm'] = 0
        df['Lsum'] = 0
        df['mag_r'] = data.magnitudes_raw
        df['mag'] = data.magnitudes_corrected
        df['ra'] = 0
        df['dec'] = 0
        return df