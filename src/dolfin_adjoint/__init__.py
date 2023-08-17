"""This just acts as an alias for fenics_adjoint.

"""
# flake8: noqa

from fenics_adjoint import *

from importlib.metadata import metadata

meta = metadata("dolfin_adjoint")

__version__ = meta["Version"]
__author__ = meta["Author"]
__license__ = meta["License"]
