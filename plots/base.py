from abc import abstractmethod

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class BasePlot:
    def __init__(self, widget, *, figsize=(8, 6)):
        self.figure = Figure(figsize=figsize)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axis = None
        self.add_axes()
        self.figure.tight_layout()

        # Finish setup

        widget.layout().addWidget(self.canvas)

    @abstractmethod
    def add_axes(self):
        """ Add axes to this plot """

    def draw(self):
        print("Drawing canvas")
        self.canvas.draw()
