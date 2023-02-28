import numpy as np
from abc import abstractmethod, ABCMeta


class PhotometryCalibration(metaclass=ABCMeta):
    def __call__(self, intensities):
        pass

    @abstractmethod
    def inverse(self, magnitudes):
        pass


class LogCalibration(PhotometryCalibration):
    def __init__(self, zero: float = 65535.0):
        self.zero = zero

    def __call__(self, intensities):
        return -2.5 * np.log10(intensities / self.zero)

    def inverse(self, magnitudes):
        pass