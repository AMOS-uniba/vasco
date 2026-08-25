"""
The Qt table models.

This package was `vasco.models`, and it held both these and the data structures the fit is built
from -- DotCollection, SensorData, Matcher. That mixture is what made importing a Matcher pull in a
window toolkit, since a parent package's __init__ runs before any module inside it, and it had to be
worked around by deleting a re-export. The data structures live in demeteor now and only the Qt
models are left, so the package says what it is.
"""
from .qcataloguemodel import CatalogueProxy, QCatalogueModel
from .qmeteormodel import QMeteorModel
from .qstarmodel import QStarModel

__all__ = ['CatalogueProxy', 'QCatalogueModel', 'QMeteorModel', 'QStarModel']
