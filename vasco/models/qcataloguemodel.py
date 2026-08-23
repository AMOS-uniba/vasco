import logging

from PyQt6.QtCore import Qt, QModelIndex, QAbstractTableModel, QSortFilterProxyModel, QVariant
from PyQt6.QtGui import QColor

log = logging.getLogger('vasco')


class QCatalogueModel(QAbstractTableModel):
    COLUMNS = ["id", "name", "dec", "ra", "alt", "az", "vmag", "use"]
    C_ID = 0
    C_NAME = 1
    C_DEC = 2
    C_RA = 3
    C_ALT = 4
    C_AZ = 5
    C_VMAG = 6
    C_VISIBLE = 7

    def __init__(self, data=None, parent=None):
        super().__init__(parent)
        self._data = [[]] if data is None else data

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = ...):
        match role:
            case Qt.ItemDataRole.DisplayRole:
                if orientation == Qt.Orientation.Horizontal:
                    return QCatalogueModel.COLUMNS[section]
            case _:
                return None

    def columnCount(self, parent=None):
        return len(self.COLUMNS)

    def rowCount(self, parent=None):
        return self._data.count

    def data(self, index: QModelIndex, role: int = ...):
        row = index.row()
        column = index.column()

        match role:
            case Qt.ItemDataRole.EditRole:
                # This is very hacky for now but I cannot get it to work with floats
                match column:
                    case self.C_ID:
                        return row
                    case self.C_NAME:
                        return self._data.name[row]
                    case self.C_DEC:
                        return self._data.dec[row]
                    case self.C_RA:
                        return self._data.ra[row]
                    case self.C_ALT:
                        return self._data.alt[row]
                    case self.C_AZ:
                        return self._data.az[row]
                    case self.C_VMAG:
                        return self._data.vmag[row]
                    case self.C_VISIBLE:
                        return self._data.mask[row]
                    case _:
                        return None
            case Qt.ItemDataRole.DisplayRole:
                match column:
                    case self.C_ID:
                        return f"{row:d}"
                    case self.C_NAME:
                        return f"{self._data.name[row]}"
                    case self.C_DEC:
                        return f"{self._data.dec[row]:.6f}°"
                    case self.C_RA:
                        return f"{self._data.ra[row]:.6f}°"
                    case self.C_ALT:
                        return f"{self._data.alt[row]:.6f}°"
                    case self.C_AZ:
                        return f"{self._data.az[row]:.6f}°"
                    case self.C_VMAG:
                        return f"{self._data.vmag[row]:.3f}m"
                    case _:
                        return None
            case Qt.ItemDataRole.TextAlignmentRole:
                if column == self.C_NAME:
                    return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            case Qt.ItemDataRole.CheckStateRole:
                match column:
                    case self.C_VISIBLE:
                        return Qt.CheckState.Checked if self._data.mask[row] else Qt.CheckState.Unchecked
            case Qt.ItemDataRole.ForegroundRole:
                if column == self.C_VISIBLE:
                    return QColor('lime') if self._data.mask[index.row()] else QColor('red')
            case Qt.ItemDataRole.BackgroundRole:
                if column == self.C_VISIBLE:
                    return QColor('green') if self._data.mask[index.row()] else QColor(128, 32, 16)

    def flags(self, index):
        flags = Qt.ItemFlag.ItemIsEnabled
        if index.column() == self.C_VISIBLE:
            flags |= Qt.ItemFlag.ItemIsUserCheckable
        return flags

    def setData(self, index, value, role: int = ...):
        log.debug("Setting data for CatalogueModel...", index.row(), index.column(), value, role)
        match role:
            case Qt.ItemDataRole.CheckStateRole:
                if index.column() == self.C_VISIBLE:
                    self._data.catalogue.mask[index.row()] = value
                    self._data.mask[index.row()] = value
                    self.dataChanged.emit(index, index)
                    return True
        return False

class CatalogueProxy(QSortFilterProxyModel):
    def lessThan(self, left, right):
        l = self.sourceModel().data(left, Qt.ItemDataRole.EditRole)
        r = self.sourceModel().data(right, Qt.ItemDataRole.EditRole)
        return bool(l < r)
