#!/usr/bin/env python


import numpy as np
from physfields import ScalarField, VectorField, Zernike


field = VectorField.from_rt(lambda r, phi: r**2 - 2, lambda r, phi: -phi)

x = np.linspace(-1, 1, 32)
y = np.linspace(-1, 1, 32)
xx, yy = np.meshgrid(x, y)

field.plot(xx, yy, file='out.pdf', colour='azimuth')


field = VectorField(lambda x, y: (-x*y, y**2))
field.plot(xx, yy, file='div.pdf', colour='div')
field.plot(xx, yy, file='rot.pdf', colour='rot')
