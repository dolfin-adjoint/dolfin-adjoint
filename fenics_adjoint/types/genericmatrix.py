import dolfin

__all__ = []
backend_genericmatrix_mul = dolfin.cpp.la.GenericMatrix.__mul__


def adjoint_genericmatrix_mul(self, other):
    out = backend_genericmatrix_mul(self, other)
    if hasattr(self, 'form') and isinstance(other, dolfin.cpp.la.GenericVector):
        if hasattr(other, 'form'):
            out.form = dolfin.action(self.form, other.form)
        elif hasattr(other, 'function'):
            if hasattr(other, 'function_factor'):
                out.form = dolfin.action(other.function_factor * self.form, other.function)
            else:
                out.form = dolfin.action(self.form, other.function)

    return out


dolfin.cpp.la.GenericMatrix.__mul__ = adjoint_genericmatrix_mul

backend_genericmatrix_ident_zeros = dolfin.cpp.la.GenericMatrix.ident_zeros


def ident_zeros(self, tol=dolfin.DOLFIN_EPS):
    backend_genericmatrix_ident_zeros(self, tol)
    self.ident_zeros_tol = tol


dolfin.cpp.la.GenericMatrix.ident_zeros = ident_zeros
