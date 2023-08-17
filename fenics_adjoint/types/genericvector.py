import dolfin
from dolfin_adjoint_common import compat
compat = compat.compat(dolfin)

__all__ = []


@staticmethod
def _ad_to_list(self):
    return compat.gather(self).tolist()


dolfin.GenericVector._ad_to_list = _ad_to_list
