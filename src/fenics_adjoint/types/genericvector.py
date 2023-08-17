import dolfin
from fenics_adjoint.utils import gather

__all__ = []


@staticmethod
def _ad_to_list(self):
    return gather(self).tolist()


dolfin.GenericVector._ad_to_list = _ad_to_list
