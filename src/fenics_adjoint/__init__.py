"""

The entire dolfin-adjoint interface should be imported with a single
call:

.. code-block:: python

  from dolfin import *
  from dolfin_adjoint import *

It is essential that the importing of the :py:mod:`dolfin_adjoint` module happen *after*
importing the :py:mod:`dolfin` module. dolfin-adjoint relies on *overloading* many of
the key functions of dolfin to achieve its degree of automation.
"""

# flake8: noqa

from pyadjoint import (Tape, set_working_tape, get_working_tape,
                       pause_annotation, continue_annotation,
                       ReducedFunctional,
                       taylor_test, taylor_to_dict,
                       compute_gradient, compute_hessian,
                       AdjFloat, Control, minimize, maximize, MinimizationProblem,
                       IPOPTSolver, ROLSolver, InequalityConstraint, EqualityConstraint,
                       MoolaOptimizationProblem, print_optimization_methods,
                       stop_annotating)
from .variational_solver import (NonlinearVariationalProblem, NonlinearVariationalSolver,
                                 LinearVariationalProblem, LinearVariationalSolver)
from .system_assembly import *
from .refine import refine
from .types import *
from .petsc_krylov_solver import PETScKrylovSolver
from .krylov_solver import KrylovSolver
from .lu_solver import LUSolver
from .newton_solver import NewtonSolver
from .ufl_constraints import UFLInequalityConstraint, UFLEqualityConstraint
from .shapead_transformations import (transfer_from_boundary,
                                      transfer_to_boundary)
from .interpolation import interpolate
from .projection import project
from .solving import solve
from .assembly import assemble, assemble_system

import pyadjoint

import sys
import dolfin


from importlib.metadata import metadata

meta = metadata("dolfin_adjoint")
__version__ = meta["Version"]
__author__ = meta["Author"]
__license__ = meta["License"]

set_working_tape(Tape())
continue_annotation()
