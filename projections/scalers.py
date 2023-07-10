import numpy as np


class Scaler:
    """ Class for scaling pixels to lengths """
    def __init__(self, scale_x: float = 1, scale_y: float = 1):
        self.scale_x = scale_x
        self.scale_y = scale_y

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return x * self.scale_x, y * self.scale_y

    def invert(self, nx: np.ndarray, ny: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return nx / self.scale_x, ny / self.scale_y