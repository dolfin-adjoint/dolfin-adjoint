from . import SolveVarFormBlock
import dolfin


class ProjectBlock(SolveVarFormBlock):
    def __init__(self, v, V, output, bcs=[], *args, **kwargs):
        mesh = kwargs.pop("mesh", None)
        if mesh is None:
            mesh = V.mesh()
        dx = dolfin.dx(mesh)
        w = dolfin.TestFunction(V)
        Pv = dolfin.TrialFunction(V)
        a = dolfin.inner(w, Pv) * dx
        L = dolfin.inner(w, v) * dx

        # Pop "function" kwarg if present.
        # This relies on the return value of project == function if given.
        kwargs.pop("function", None)

        super(ProjectBlock, self).__init__(a == L, output, bcs, *args, **kwargs)
