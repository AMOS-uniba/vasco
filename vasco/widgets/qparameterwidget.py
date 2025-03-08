from typing import Callable
from PyQt6.QtWidgets import QWidget

from widgets.qparameterwidget_ui import Ui_Form


class QParameterWidget(QWidget, Ui_Form):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setupUi(self)

        self.display_to_true = lambda x: x
        self.true_to_display = lambda x: x

    def setup(self,
              *,
              title: str = "(unknown)",
              symbol: str = "?",
              unit: str = "",
              minimum: float = 0,
              maximum: float = 1,
              step: float = 0.001,
              decimals: int = 6,
              initial_value: float = 0,
              display_to_true: Callable[[float], float] = lambda x: x,
              true_to_display: Callable[[float], float] = lambda x: x):
        self.lb_title.setText(title)
        self.lb_symbol.setText(symbol)
        self.lb_symbol.setBuddy(self.dsb_value)
        self.lb_unit.setText(unit)
        self.dsb_value.setMinimum(minimum)
        self.dsb_value.setMaximum(maximum)
        self.dsb_value.setSingleStep(step)
        self.dsb_value.setDecimals(decimals)
        self.dsb_value.setValue(initial_value)

        self.display_to_true = display_to_true
        self.true_to_display = true_to_display

    def is_checked(self) -> bool:
        return self.cb_enabled.isChecked()

    @property
    def display_value(self) -> float:
        return self.dsb_value.value()

    def set_display_value(self, new_value: float):
        self.dsb_value.setValue(new_value)

    @property
    def true_value(self) -> float:
        return float(self.display_to_true(self.display_value))

    def set_true_value(self, new_value: float):
        self.set_display_value(self.true_to_display(new_value))
