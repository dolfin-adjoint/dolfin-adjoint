import numpy
import dolfin


def constant_from_values(constant, values=None):
    """Returns a new Constant with `constant.values()` while preserving `constant.ufl_shape`.

    If the optional argument `values` is provided, then `values` will be the values of the
    new Constant instead, while still preserving the ufl_shape of `constant`.

    Args:
        constant: A constant with the ufl_shape to preserve.
        values (numpy.array): An optional argument to use instead of constant.values().

    Returns:
        Constant: The created Constant of the same type as `constant`.

    """
    values = constant.values() if values is None else values
    return type(constant)(numpy.reshape(values, constant.ufl_shape))


def function_from_vector(V, vector):
    """Create a new Function from a vector.

    :arg V: The function space
    :arg vector: The vector data.
    """
    if isinstance(vector, dolfin.cpp.la.PETScVector)\
            or isinstance(vector, dolfin.cpp.la.Vector):
        pass
    elif not isinstance(vector, dolfin.Vector):
        # If vector is a fenics_adjoint.Function, which does not inherit
        # dolfin.cpp.function.Function with pybind11
        vector = vector._cpp_object
    r = dolfin.Function(V)
    r.vector()[:] = vector
    return r


def extract_subfunction(u, V):
    component = V.component()
    r = u
    for idx in component:
        r = r.sub(int(idx))
    return r


def as_backend_type(A):
    out = dolfin.as_backend_type(A)
    out._ad_original_ref = A
    return out


def create_function(*args, **kwargs):
    """Initialises a fenics_adjoint.Function object and returns it."""
    from fenics_adjoint import Function
    return Function(*args, **kwargs)


MatrixTypes = (dolfin.cpp.la.Matrix, dolfin.cpp.la.GenericMatrix)


def extract_mesh_from_form(form):
    """Takes in a form and extracts a mesh which can be used to construct function spaces.

    Dolfin only accepts backend.cpp.mesh.Mesh types for function spaces, while firedrake use ufl.Mesh.

    Args:
        form (ufl.Form): Form to extract mesh from

    Returns:
        dolfin.Mesh: The extracted mesh

    """
    return form.ufl_domain().ufl_cargo()


def gather(vec):
    import numpy
    if isinstance(vec, dolfin.cpp.function.Function):
        vec = vec.vector()

    if isinstance(vec, dolfin.cpp.la.GenericVector):
        arr = vec.gather(numpy.arange(vec.size(), dtype='I'))
    elif isinstance(vec, list):
        return list(map(gather, vec))
    else:
        arr = vec  # Assume it's a gathered numpy array already

    return arr


def linalg_solve(A, x, b, *args, **kwargs):
    """Linear system solve that has a firedrake compatible interface.

    Throws away kwargs and uses b.vector() as RHS if
    b is not a GenericVector instance.

    """
    if not isinstance(b, dolfin.GenericVector):
        b = b.vector()
    return dolfin.solve(A, x, b, *args)


def create_function(*args, **kwargs):
    """Initialises a fenics_adjoint.Function object and returns it."""
    from fenics_adjoint import Function
    return Function(*args, **kwargs)
