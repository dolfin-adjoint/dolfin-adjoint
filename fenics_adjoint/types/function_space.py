import dolfin
from dolfin_adjoint_common import compat
compat = compat.compat(dolfin)


extract_subfunction = compat.extract_subfunction

__all__ = ["extract_subfunction"]
