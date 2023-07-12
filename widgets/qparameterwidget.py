from PyQt6.QtWidgets import QWidget

from widgets.qparameterwidget_ui import Ui_Form


class QParameterWidget(QWidget, Ui_Form):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setupUi(self)

        self.lb_unit.setText("penis")