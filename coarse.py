#!/usr/bin/env python

import sys
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt

from typing import Tuple, Type, Optional
from utilities import by_azimuth, polar_to_cart

import PyQt6
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.uic import loadUi

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

from shifters import OpticalAxisShifter, EllipticShifter
from transformers import LinearTransformer, ExponentialTransformer, BiexponentialTransformer
from projections import Projection, BorovickaProjection

from main_ui import Ui_MainWindow

mpl.use('Qt5Agg')

COUNT = 100


class Fitter():
    def __init__(self):
        pass

    def __call__(self, xy: Tuple[np.ndarray, np.ndarray], za: Tuple[np.ndarray, np.ndarray], cls: Type[Projection], *, params: Optional[dict]=None) -> Projection:
        """
            xy      a 2-tuple of x and y coordinates on the sensor
            za      a 2-tuple of z and a coordinates in the sky catalogue
            cls     a subclass of Projection that is used to transform xy onto za
            Returns an instance of cls with parameters set to values that result in minimal deviation
        """
        return cls(params)


class Vascop():
    """ Virtual All-Sky CorrectOr Plate """
    def __init__(self):
        pass
#        self.argparser = argparse.ArgumentParser("Virtual all-sky corrector plate")
#        self.argparser.add_argument('infile', type=argparse.FileType('r'), help="input file")
#        self.argparser.add_argument('outdir', action=argparser.WriteableDir, help="output directory")
#        self.argparser.add_argument('method', type=str, choices=Corrector.METHODS)
#        self.args = self.argparser.parse_args()
#        self.outdir = Path(self.args.outdir)

    def load(self, filename):
        df = pd.read_csv(filename, sep='\t', header=0, nrows=500)
        df['a_cat_rad'] = np.radians(df['acat'])
        df['z_cat_rad'] = df['zcat'] / 90
        df['x_cat'], df['y_cat'] = polar_to_cart(df['z_cat_rad'], df['a_cat_rad'])
        df['a_com_rad'] = np.radians(df['acom'])
        df['z_com_rad'] = df['zcom'] / 90
        df['x_com'], df['y_com'] = polar_to_cart(df['z_com_rad'], df['a_com_rad'])
        df['dx'], df['dy'] = df['x_cat'] - df['x_com'], df['y_cat'] - df['y_com']
        self.data = df
        self.points = self.data[['x_cat', 'y_cat']].to_numpy()
        self.values = self.data[['dx', 'dy']].to_numpy()


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.connectSignalSlots()

        self.vascop = Vascop()
        self.vascop.load('data/borr-01.tsv')

        self.sensorFigure = Figure(figsize=(5, 5))
        self.sensorCanvas = FigureCanvas(self.sensorFigure)
        self.sensorAxis = self.sensorFigure.add_subplot()
        self.sensorFigure.tight_layout()

        self.skyFigure = Figure(figsize=(5, 5))
        self.skyCanvas = FigureCanvas(self.skyFigure)
        self.skyAxis = self.skyFigure.add_subplot(projection='polar')
        self.skyFigure.tight_layout()

        self.sensorAxis.set_xlim([-1, 1])
        self.sensorAxis.set_ylim([-1, 1])
        self.sensorAxis.grid()
        self.sensorAxis.set_aspect('equal')

        self.skyAxis.set_ylim([0, 90])
        self.skyAxis.set_theta_offset(3 * np.pi / 2)
        self.sensorScatter = self.sensorAxis.scatter([0], [0])
        self.skyScatter = self.skyAxis.scatter([0], [0])

        self.sensorScatter.set_offsets(np.stack((self.vascop.points[:, 0], self.vascop.points[:, 1]), axis=1))
        self.sensorCanvas.draw()

        self.plot()

        self.gb_plots.layout().addWidget(self.sensorCanvas)
        self.gb_plots.layout().addWidget(self.skyCanvas)

    def connectSignalSlots(self):
        self.dsb_x0.valueChanged.connect(self.plot)
        self.dsb_y0.valueChanged.connect(self.plot)
        self.dsb_a0.valueChanged.connect(self.plot)
        self.dsb_V.valueChanged.connect(self.plot)
        self.dsb_S.valueChanged.connect(self.plot)
        self.dsb_D.valueChanged.connect(self.plot)
        self.dsb_P.valueChanged.connect(self.plot)
        self.dsb_Q.valueChanged.connect(self.plot)
        self.dsb_A.valueChanged.connect(self.plot)
        self.dsb_F.valueChanged.connect(self.plot)
        self.dsb_eps.valueChanged.connect(self.plot)
        self.dsb_E.valueChanged.connect(self.plot)

    def plot(self):
        proj = BorovickaProjection(
            x0=self.dsb_x0.value(),
            y0=self.dsb_y0.value(),
            a0=np.radians(self.dsb_a0.value()),
            V=self.dsb_V.value(),
            S=self.dsb_S.value(),
            D=self.dsb_D.value(),
            P=self.dsb_P.value(),
            Q=self.dsb_Q.value(),
            A=self.dsb_A.value(),
            F=np.radians(self.dsb_F.value()),
            epsilon=np.radians(self.dsb_eps.value()),
            E=np.radians(self.dsb_E.value()),
        )

#        print(proj)

        x, y = self.vascop.points[:, 0], self.vascop.points[:, 1]
        z, a = proj(x, y)
        z = np.degrees(z)

        self.skyScatter.set_offsets(np.stack((a, z), axis=1))
        self.skyCanvas.draw()


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
