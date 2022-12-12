import scipy as sp


class VascoPaired(Vasco):
    def __init__(self, sensor, catalogue):
        self.sensor = sensor
        self.catalogue = catalogue

    def compute_distances(self, stars, catalogue):
        catalogue = np.radians(catalogue)
        stars[:, 0] = np.pi / 2 - stars[:, 0]
        return distance(stars, catalogue)

    def func(self, x, *args):
        projection = BorovickaProjection(*x)
        return self.calculate_error(BorovickaProjection, *args)

    def minimize(self, location, time, x0=(0, 0, 0, 0, 0, np.pi / 2, 0, 0, 0, 0, 0, 0)):
        result = sp.optimize.minimize(self.func, x0, args=(location, time), method='Nelder-Mead')
        return result
