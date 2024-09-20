import math
import numpy as np

from .base import Projection


class ZenithShifter(Projection):
    """
    ZenithShifter is a spherical -> spherical projection that shifts the zenith to a different
    position at true zenith distance `epsilon` and rotated such that true zenith is at 180°.
    """
    def __init__(self, epsilon: float = 0, E: float = 0):
        super().__init__()
        self.epsilon = epsilon
        self.E = E

    def __call__(self, u, b):
        """
        Parameters
        ----------
        u : Union[float, np.ndarray]    radial distance from origin in camera coordinates
        b : Union[float, np.ndarray]    azimuth in camera coordinates

        Returns
        -------
        Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]
        z : Union[float, np.ndarray]    zenith distance in sky coordinates
        a : Union[float, np.ndarray]    azimuth in sky coordinates
        """
        if abs(self.epsilon) < 1e-14:  # for tiny epsilon there is no displacement
            z = u  # and we are able to calculate the coordinates immediately
            a = self.E + b
        else:
            cosz = np.cos(u) * np.cos(self.epsilon) - np.sin(u) * np.sin(self.epsilon) * np.cos(b)
            sna = np.sin(b) * np.sin(u)
            cna = (np.cos(u) - np.cos(self.epsilon) * cosz) / np.sin(self.epsilon)
            z = np.arccos(cosz)
            a = self.E + np.arctan2(sna, cna)

        return z, np.mod(a, math.tau)  # wrap around to [0, 2pi)

    def invert(self, z, a):
        if abs(self.epsilon) < 1e-14:
            u = z
            b = a - self.E
        else:
            cosu = np.cos(z) * np.cos(self.epsilon) + np.sin(z) * np.sin(self.epsilon) * np.cos(a - self.E)
            sna = np.sin(a - self.E) * np.sin(z)
            cna = -(np.cos(z) - np.cos(self.epsilon) * cosu) / np.sin(self.epsilon)
            u = np.arccos(cosu)
            b = np.arctan2(sna, cna)

        return u, np.mod(b, math.tau)  # wrap around to [0, 2pi)

    def __str__(self):
        return f"<{self.__class__.__name__} epsilon={self.epsilon} E={self.E}>"

    def as_dict(self):
        return dict(
            epsilon=float(self.epsilon),
            E=float(self.E),
        )