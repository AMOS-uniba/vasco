import numpy as np

from .base import Projection


class ZenithShifter(Projection):
    def __init__(self, epsilon: float = 0, E: float = 0):
        super().__init__()
        self.epsilon = epsilon
        self.E = E

    def __call__(self, u, b):
        if abs(self.epsilon) < 1e-14:  # for tiny epsilon there is no displacement
            z = u  # and we are able to calculate the coordinates immediately
            a = self.E + b
        else:
            cosz = np.cos(u) * np.cos(self.epsilon) - np.sin(u) * np.sin(self.epsilon) * np.cos(b)
            sna = np.sin(b) * np.sin(u)
            cna = (np.cos(u) - np.cos(self.epsilon) * cosz) / np.sin(self.epsilon)
            z = np.arccos(cosz)
            a = self.E + np.arctan2(sna, cna)

        a = np.fmod(a + 2 * np.pi, 2 * np.pi)  # wrap around to [0, 2pi)
        return z, a

    def invert(self, z, a):
        if abs(self.epsilon) < 1e-14:
            u = z
            b = a - self.E
        else:
            cosu = np.cos(z) * np.cos(self.epsilon) - np.sin(z) * np.sin(self.epsilon) * np.cos(a - self.E)
            sna = np.sin(a - self.E) * np.sin(z)
            cna = (np.cos(z) - np.cos(self.epsilon) * cosu) / np.sin(self.epsilon)
            u = np.arccos(cosu)
            b = np.arctan2(sna, cna)

        b = np.fmod(b + 2 * np.pi, 2 * np.pi)
        return u, b

