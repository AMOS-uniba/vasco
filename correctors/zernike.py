import numpy as np

from .base import BaseCorrector
from . import kernels
from .kernelsmoother import KernelSmoother

from physfields import ZernikeVector, VectorField


np.set_printoptions(edgeitems=30999, linewidth=100000, formatter={'float': lambda x: f"{x:8.6f}"})


class ZernikeFitter():
    @staticmethod
    def project(field, basic, nodes):
        sampled = basic.eval(nodes)
        count = sampled.count() / 2
        return np.sum(field * sampled) / count


    def __call__(self, nodes, field, order=7):
        result = np.zeros_like(field)

        disk = np.sum(np.square(nodes), axis=1) > 1
        mask = np.stack((disk, disk), axis=1)

        for n in range(1, order + 1):
            for l in range(-n, n + 1, 2):
                for r in [None] if abs(l) == n else [True, False]:
                    basic = ZernikeVector(n, l, r)
                    evaluated = basic.eval(nodes)
                    proj = np.sum(evaluated * field) * evaluated / (nodes.count() / 2)
                    result += proj

        return result


class ZernikeExpander(BaseCorrector):
    def __init__(self, points, values, kernel=kernels.nexp, bandwidth=1.0, **kwargs):
        super().__init__(points, values)
        self.bandwidth = bandwidth

    def __call__(self, nodes, order=7, bandwidth=1.0):
        ks = KernelSmoother(self.points, self.values, bandwidth=self.bandwidth)
        uv = ks(nodes)
        return ZernikeFitter()(nodes, uv, order)


    def test(self, nodes):
        for k, v in self.v.items():
            for l, w in self.v.items():
                n = nodes.T
                vu, vv = v(n[:, 0], n[:, 1])
                wu, wv = w(n[:, 0], n[:, 1])

                disk = np.sum(np.square(n), axis=1) > 1
                vu = np.ma.masked_where(disk, vu)
                vv = np.ma.masked_where(disk, vv)
                wu = np.ma.masked_where(disk, wu)
                wv = np.ma.masked_where(disk, wv)
                dp = np.sum(vu * wu + vv * wv) / vu.count()

                if abs(dp) > 1e-6:
                    print(f"{str(k):>15} × {str(l):>15} = {dp:10.6f}")
