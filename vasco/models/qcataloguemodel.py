from PyQt6.QtCore import Qt, QModelIndex, QAbstractTableModel
from PyQt6.QtGui import QIcon, QColor
from PyQt6.QtWidgets import QWidget, QStyle


class QCatalogueModel(QAbstractTableModel):
    def __init__(self, data=None, parent=None):
        super().__init__(parent)
        self._data = [[]] if data is None else data

    def headerData(self, section: int, orientation: Qt.Orientation, role: int):
        match role:
            case Qt.ItemDataRole.DisplayRole:
                if orientation == Qt.Orientation.Horizontal:
                    return ["x [px]", "y [px]", "x [mm]", "y [mm]", "alt", "az", "mask"][section]
            case _:
                return None


    def columnCount(self, parent=None):
        return 7

    def rowCount(self, parent=None):
        return self._data.count

    def data(self, index: QModelIndex, role: int):
        match role:
            case Qt.ItemDataRole.DisplayRole:
                row = index.row()
                match index.column():
                    case 0:
                        return f"{self._data.px[row]:.3f}"
                    case 1:
                        return f"{self._data.py[row]:.3f}"
                    case 2:
                        return f"{self._data.x[row]:.6f}"
                    case 3:
                        return f"{self._data.y[row]:.6f}"
                    case 4:
                        return f"{self._data.alt[row]:.6f}°"
                    case 5:
                        return f"{self._data.az[row]:.6f}°"
                    case 6:
                        return f"{'\u2714' if self._data.mask[row] else '\u274C'}"
                    case _:
                        return None
            case Qt.ItemDataRole.TextAlignmentRole:
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            case Qt.ItemDataRole.ForegroundRole:
                if index.column() == 6:
                    return QColor('green') if self._data.mask[index.row()] else QColor('red')
