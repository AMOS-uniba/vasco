from PyQt6.QtCore import Qt, QModelIndex, QAbstractTableModel
from PyQt6.QtGui import QColor


class QCatalogueModel(QAbstractTableModel):
    COLUMNS = ["id", "dec", "ra", "alt", "az", "vmag", "visible"]

    def __init__(self, data=None, parent=None):
        super().__init__(parent)
        self._data = [[]] if data is None else data

    def headerData(self, section: int, orientation: Qt.Orientation, role: int):
        match role:
            case Qt.ItemDataRole.DisplayRole:
                if orientation == Qt.Orientation.Horizontal:
                    return QCatalogueModel.COLUMNS[section]
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
                        return f"{row}"
                    case 1:
                        return f"{self._data.dec[row]:.6f}°"
                    case 2:
                        return f"{self._data.ra[row]:.6f}°"
                    case 3:
                        return f"{self._data.alt[row]:.6f}°"
                    case 4:
                        return f"{self._data.az[row]:.6f}°"
                    case 5:
                        return f"{self._data.vmag[row]:.3f}m"
                    case 6:
                        return '\u2714' if self._data.mask[row] else '\u274C'
                    case _:
                        return None
            case Qt.ItemDataRole.TextAlignmentRole:
                match index.column():
                    case 6:
                        return Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            case Qt.ItemDataRole.ForegroundRole:
                if index.column() == 6:
                    return QColor('green') if self._data.mask[index.row()] else QColor('red')
