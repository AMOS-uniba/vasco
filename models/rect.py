import numpy as np

from projections.shifters import ScalingShifter


class Rect:
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.xcen = (xmax + xmin) / 2
        self.ycen = (ymax + ymin) / 2
        self.r = np.sqrt((ymax - ymin)**2 + (xmax - xmin)**2) / 2
        self.shifter = ScalingShifter(x0=self.xcen, y0=self.ycen, scale=1 / self.r)

    def to_unit(self, data):
        """ data: np.ndarray(N, 2) """
        out = np.zeros_like(data)
        out[:, 0] = (data[:, 0] - self.xcen) / self.r
        out[:, 1] = (data[:, 1] - self.ycen) / self.r
        return out


