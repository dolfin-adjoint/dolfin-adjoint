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

        # Solver_type is specific for project, so we rename it
        solver_type = kwargs.pop("solver_type", None)
        prec_type = kwargs.pop("preconditioner_type", None)
        forward_kwargs = kwargs.pop("forward_kwargs", {})
        forward_solver_params = forward_kwargs.pop("solver_parameters", {})
        forward_solver_params.update({"linear_solver": solver_type, "preconditioner": prec_type})
        forward_kwargs.update({"solver_parameters": forward_solver_params})
        kwargs["forward_kwargs"] = forward_kwargs
        super(ProjectBlock, self).__init__(a == L, output, bcs, *args, **kwargs)
