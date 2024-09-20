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
