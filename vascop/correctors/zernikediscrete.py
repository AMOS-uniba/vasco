import numpy as np

from matplotlib import pyplot as plt

from .base import BaseCorrector
from physfields import ZernikeVector, VectorField


class ZernikeDiscreteExpander(BaseCorrector):
    vs = {}
    """ Does not work as expected """

    for n in range(1, 3):
        for l in range(-n, n + 1, 2):
            if abs(l) == n:
                vs[n, l] = ZernikeVector(n, l)
            else:
                vs[n, l, False] = ZernikeVector(n, l, False)
                vs[n, l, True] = ZernikeVector(n, l, True)

    def __init__(self, points, values):
        super().__init__(points, values)

    def __call__(self, nodes):
        for k, v in self.vs.items():
            fig, ax = plt.subplots(figsize=(10, 10))
            fig.tight_layout()
            ax.set_facecolor('white')
            ax.set_aspect('equal')
            ax.quiver(self.points[:, 0], self.points[:, 1], self.values[:, 0], self.values[:, 1], scale=0.2, width=0.0025, color='blue')

            vu, vv = v(self.points[:, 0], self.points[:, 1])

            # Calculate the vector field at positions of measurements
            field = np.stack((vu, vv), axis=0).T
            # Calculate the dot product of the field with the measurements
            dot = np.sum(field * self.values)
            print(f"Sum of dot products with {k} is {dot:8.6f}")
            # Calculate the norm of the field at positions of measurements
            length = np.sum(field * field, axis=1)
            # Projection of vec(a) onto vec(b) is (a.a / b.b) . b
            projected = np.expand_dims((dot / length) / field.shape[0], 1)
            mean = np.mean(projected)
            print(f"Mean is {mean}")

            self.values -= mean * field

            x = np.linspace(-1, 1, 30)

            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            v.plot_data(ax, *np.meshgrid(x, x), mask=VectorField.UnitDisk, file=f'out/{k}.png')
            ax.scatter(self.points[:, 0], self.points[:, 1], s=3, color='red')
            ax.quiver(self.points[:, 0], self.points[:, 1], self.values[:, 0], self.values[:, 1], scale=0.2, width=0.0025, color='red')
            plt.show()

        f = VectorField(lambda x, y: np.sum(np.stack([v(x, y) for v in z.values()]), axis=0))

        return f(nodes[0], nodes[1])

