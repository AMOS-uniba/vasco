from PyQt6.QtCore import Qt, QVariant, QModelIndex, QAbstractTableModel


class QMeteorModel(QAbstractTableModel):
    def __init__(self, data=None, parent=None):
        super().__init__(parent)
        self._data = [[]] if data is None else data

    def headerData(self, section: int, orientation: Qt.Orientation, role: int):
        match role:
            case Qt.ItemDataRole.DisplayRole:
                if orientation == Qt.Orientation.Horizontal:
                    return\
                        ["fno",
                         "z raw", "a raw",
                         "z corrected", "a corrected",
                         "corr x / mm", "corr y / mm",
                         "corr total",
                         "mag raw", "mag corr",
                         ][section]
            case _:
                return QVariant()

    def columnCount(self, parent=None):
        return 10

    def rowCount(self, parent=None):
        return len(self._data)

    def data(self, index: QModelIndex, role: int):
        match role:
            case Qt.ItemDataRole.DisplayRole:
                row = index.row()
                match index.column():
                    case 0:
                        return f"{self._data.fnos[row]:d}"
                    case 1:
                        return f"{self._data.position_raw.alt[row].value:.6f}°"
                    case 2:
                        return f"{self._data.position_raw.az[row].value:.6f}°"
                    case 3:
                        return f"{self._data.position_corrected.alt[row].value:.6f}°"
                    case 4:
                        return f"{self._data.position_corrected.az[row].value:.6f}°"
                    case 5:
                        return f"{self._data.positions_correction_xy[row, 0]:.6f}"
                    case 6:
                        return f"{self._data.positions_correction_xy[row, 1]:.6f}"
                    case 7:
                        return f"{self._data.positions_correction_angle[row].degree:.6f}°"
                    case 8:
                        return f"{self._data.magnitudes_corrected[row]:.6f}"
                    case 9:
                        return f"{self._data.magnitudes_correction[row]:.6f}"
                    case _:
                        return QVariant()
            case Qt.ItemDataRole.TextAlignmentRole:
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
