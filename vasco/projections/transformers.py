import numpy as np
import scipy as sp


class RadialTransformer:
    """ Class for transforming radial distances in all-sky projections """

    def __call__(self, r):
        raise NotImplementedError("Radial transformers must implement __call__(r: np.ndarray) -> np.ndarray")

    def fprime(self, u):
        """ Derivative function for the Newton method, not implemented in the abstract base class """
        raise NotImplementedError("Radial transformers should implement df/dr as fprime(u: np.ndarray) -> np.ndarray")

    def invert(self, u):
        """ Numerically approximate the inverse function using the Newton method """
        return sp.optimize.newton(lambda r: self.__call__(r) - u, np.zeros_like(u), self.fprime)


class LinearTransformer(RadialTransformer):
    """ Linear radial transform, u = Vr """

    def __init__(self, linear: float = 1):
        self.linear = linear  # radial stretch, linear coefficient

    def __call__(self, r):
        return self.linear * r

    def fprime(self, r):
        """ du / dr = V """
        return self.linear * np.ones_like(r)

    def as_dict(self):
        return dict(
            V=float(self.linear),
        )


class ExponentialTransformer(LinearTransformer):
    """ Linear + exponential radial correction, u = Vr + S(e^(Dr) - 1) """

    def __init__(self, linear: float = 1, lin_coef: float = 0, lin_exp: float = 0):
        super().__init__(linear)
        self.lin_coef = lin_coef  # radial stretch, exponential term, linear coefficient
        self.lin_exp = lin_exp  # radial stretch, exponential term, exponent coefficient

    def __call__(self, r):
        return super().__call__(r) + self.lin_coef * (np.exp(self.lin_exp * r) - 1)

    def fprime(self, r):
        """ du/dr = V + SDe^(Dr) """
        return super().fprime(r) + self.lin_coef * self.lin_exp * np.exp(self.lin_exp * r)

    def as_dict(self):
        return super().as_dict() | dict(
            S=float(self.lin_coef),
            D=float(self.lin_exp),
        )


class BiexponentialTransformer(ExponentialTransformer):
    """ Bi-exponential radial fitting procedure, u = Vr + S(e^(Dr) - 1) + P(e^(Qr^2) - 1) """

    def __init__(self, linear: float = 0,
                 lin_coef: float = 0, lin_exp: float = 0,
                 quad_coef: float = 0, quad_exp: float = 0):
        super().__init__(linear, lin_coef, lin_exp)
        self.quad_coef = quad_coef  # radial stretch, square-exponential term, linear coefficient
        self.quad_exp = quad_exp    # radial stretch, square-exponential term, exponent coefficient

    def __call__(self, r):
        return super().__call__(r) + self.quad_coef * (np.exp(self.quad_exp * r * r) - 1)

    def fprime(self, r):
        """ du/dr = V + SDe^(Dr) + 2 PQr e^(Qr^2) """
        return super().fprime(r) + 2 * self.quad_coef * self.quad_exp * r * np.exp(self.quad_exp * r * r)

    def __str__(self):
        return f"<{self.__class__.__name__} V={self.linear} S={self.lin_coef} " \
               f"D={self.lin_exp} P={self.quad_coef} Q={self.quad_exp}>"

    def as_dict(self):
        return super().as_dict() | dict(
            P=float(self.quad_coef),
            Q=float(self.quad_exp),
        )
