import dolfin


__set_nullspace = dolfin.cpp.la.PETScMatrix.set_nullspace
def set_nullspace(self, nullspace):
    self._ad_original_ref._ad_nullspace = nullspace
    __set_nullspace(self, nullspace)
dolfin.cpp.la.PETScMatrix.set_nullspace = set_nullspace

__set_near_nullspace = dolfin.cpp.la.PETScMatrix.set_near_nullspace
def set_near_nullspace(self, near_nullspace):
    self._ad_original_ref._ad_near_nullspace = near_nullspace
    __set_near_nullspace(self, near_nullspace)
dolfin.cpp.la.PETScMatrix.set_near_nullspace = set_near_nullspace

class VectorSpaceBasis(dolfin.VectorSpaceBasis):
    def __init__(self, *args, **kwargs):
        super(VectorSpaceBasis, self).__init__(*args, **kwargs)
        self._ad_orthogonalized = False

    def orthogonalize(self, vector):
        dolfin.VectorSpaceBasis.orthogonalize(self, vector)
        self._ad_orthogonalized = True
