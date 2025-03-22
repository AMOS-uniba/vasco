from PyQt6.QtCore import Qt, QModelIndex, QAbstractTableModel
from PyQt6.QtGui import QColor


class QStarModel(QAbstractTableModel):
    COLUMNS = ["#", "x [px]", "y [px]", "x [mm]", "y [mm]",
               "alt", "az", "used", "star",
               "alt error", "az error", "total error"]

    def __init__(self, data=None, parent=None):
        super().__init__(parent)
        self._data = [[]] if data is None else data

    def headerData(self, section: int, orientation: Qt.Orientation, role: int):
        match role:
            case Qt.ItemDataRole.DisplayRole:
                if orientation == Qt.Orientation.Horizontal:
                    return QStarModel.COLUMNS[section]
            case _:
                return None


    def columnCount(self, parent=None):
        return 12

    def rowCount(self, parent=None):
        return self._data.count

    def data(self, index: QModelIndex, role: int):
        match role:
            case Qt.ItemDataRole.DisplayRole:
                row = index.row()
                match index.column():
                    case 0:
                        return f"{row}"
                    case 1:
                        return f"{self._data.px[row]:.3f}"
                    case 2:
                        return f"{self._data.py[row]:.3f}"
                    case 3:
                        return f"{self._data.x[row]:.6f}"
                    case 4:
                        return f"{self._data.y[row]:.6f}"
                    case 5:
                        return f"{90 - self._data.alt[row]:.6f}°"
                    case 6:
                        return f"{self._data.az[row]:.6f}°"
                    case 7:
                        return '\u2714' if self._data.mask[row] else '\u274C'
                    case 8:
                        if self._data.star is None:
                            return "-"
                        else:
                            return f"{self._data.star[row]}"
                    case 9:
                        return f"{self._data.vector_errors[row, 0]:.6f}°"
                    case 10:
                        return f"{self._data.vector_errors[row, 1]:.6f}°"
                    case 11:
                        return f"{self._data.scalar_errors[row]:.6f}°"
                    case _:
                        return None
            case Qt.ItemDataRole.TextAlignmentRole:
                match index.column():
                    case 7:
                        return Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
                    case _:
                        return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            case Qt.ItemDataRole.ForegroundRole:
                match index.column():
                    case 7:
                        return QColor('green') if self._data.mask[index.row()] else QColor('red')
                    case 11:
                        if (value := self._data.scalar_errors[index.row()]) < 0.2:
                            h = int((0.2 - value) * 600)
                        else:
                            h = 0
                        return QColor.fromHsl(h, 255, 128)
