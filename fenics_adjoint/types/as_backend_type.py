import fenics


def as_backend_type(A):
    out = fenics.as_backend_type(A)
    out._ad_original_ref = A
    return out


__set_nullspace = fenics.cpp.la.PETScMatrix.set_nullspace


def set_nullspace(self, nullspace):
    self._ad_original_ref._ad_nullspace = nullspace
    __set_nullspace(self, nullspace)


fenics.cpp.la.PETScMatrix.set_nullspace = set_nullspace


class VectorSpaceBasis(fenics.VectorSpaceBasis):
    def __init__(self, *args, **kwargs):
        super(VectorSpaceBasis, self).__init__(*args, **kwargs)
        self._ad_orthogonalized = False

    def orthogonalize(self, vector):
        fenics.VectorSpaceBasis.orthogonalize(self, vector)
        self._ad_orthogonalized = True
