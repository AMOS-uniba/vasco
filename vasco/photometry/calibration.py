import numpy as np
from abc import abstractmethod, ABCMeta


class Calibration(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, intensities: np.ndarray[float]) -> np.ndarray[float]:
        """ Transform raw intensities to magnitudes """

    @abstractmethod
    def inverse(self, magnitudes: np.ndarray[float]) -> np.ndarray[float]:
        """ Transform magnitudes to raw intensities """


class LogCalibration(Calibration):
    def __init__(self, zero: float = 10000):
        self.zero = zero

    def __call__(self, intensities: np.ndarray[float]) -> np.ndarray[float]:
        return -2.5 * np.log10(intensities / self.zero)

    def inverse(self, magnitudes: np.ndarray[float]) -> np.ndarray[float]:
        return self.zero * 10**(-0.4 * magnitudes)
