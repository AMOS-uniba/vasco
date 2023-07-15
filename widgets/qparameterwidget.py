from typing import Callable
from PyQt6.QtWidgets import QWidget

from widgets.qparameterwidget_ui import Ui_Form


class QParameterWidget(QWidget, Ui_Form):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setupUi(self)

        self.inner_function = lambda x: x
        self.input_function = lambda x: x

    def setup(self,
              *,
              title: str = "(unknown)",
              symbol: str = "?",
              unit: str = "",
              minimum: float = 0,
              maximum: float = 1,
              step: float = 0.001,
              decimals: int = 6,
              inner_function: Callable[[float], float] = lambda x: x,
              input_function: Callable[[float], float] = lambda x: x):
        self.lb_title.setText(title)
        self.lb_symbol.setText(symbol)
        self.lb_symbol.setBuddy(self.dsb_value)
        self.lb_unit.setText(unit)
        self.dsb_value.setMinimum(minimum)
        self.dsb_value.setMaximum(maximum)
        self.dsb_value.setSingleStep(step)
        self.dsb_value.setDecimals(decimals)

        self.inner_function = inner_function
        self.input_function = input_function

    def is_checked(self) -> bool:
        return self.cb_enabled.isChecked()

    @property
    def value(self) -> float:
        return self.dsb_value.value()

    def set_value(self, new_value: float):
        self.dsb_value.setValue(new_value)

    def inner_value(self) -> float:
        return float(self.inner_function(self.value))

    def set_from_gui(self, new_value: float):
        self.set_value(self.input_function(new_value))