import numpy as np


def gaussian(x, mu, sigma):
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((x - mu)**2) / (2 * sigma**2))


def nexp(x):
    return np.exp(-x)


def ugauss(x):
    return gaussian(x, 0, 1)


def inv(s=1e-6):
    def fun(x):
        return 1 / (x + s)
    return fun


def epanechnikov(x):
    return np.where(x > 1, 0, 0.75 * (1 - x**2))
