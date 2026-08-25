"""
The command-line faces of vasco.

`gui` is the window; `fit` is the same fitting code with nobody driving it, for a server that has a
plate, a list of reference dots and no keyboard. Kept apart from the fitting code itself so that
importing vasco costs nothing but numpy, scipy and astropy -- see vasco/models/__init__.py.
"""
