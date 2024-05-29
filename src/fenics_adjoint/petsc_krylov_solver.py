import dolfin

from pyadjoint.tape import annotate_tape, get_working_tape


from fenics_adjoint.blocks import PETScKrylovSolveBlock, PETScKrylovSolveBlockHelper
from fenics_adjoint.utils import MatrixTypes


class PETScKrylovSolver(dolfin.PETScKrylovSolver):
    def __init__(self, *args, **kwargs):
        self.ad_block_tag = kwargs.pop("ad_block_tag", None)
        dolfin.PETScKrylovSolver.__init__(self, *args, **kwargs)

        A = kwargs.pop("A", None)
        method = kwargs.pop("method", "default")
        preconditioner = kwargs.pop("preconditioner", "default")

        next_arg_idx = 0
        if len(args) > 0 and isinstance(args[0], MatrixTypes):
            A = args[0]
            next_arg_idx = 1
        elif len(args) > 1 and isinstance(args[1], MatrixTypes):
            A = args[1]
            next_arg_idx = 2

        if len(args) > next_arg_idx and isinstance(args[next_arg_idx], str):
            method = args[next_arg_idx]
            next_arg_idx += 1
            if len(args) > next_arg_idx and isinstance(args[next_arg_idx], str):
                preconditioner = args[next_arg_idx]

        self.operator = A
        self.pc_operator = None
        self.method = method
        self.preconditioner = preconditioner
        self.solver_parameters = {}
        self.block_helper = PETScKrylovSolveBlockHelper()
        self._ad_nullspace = None
        if hasattr(self.operator, "_ad_nullspace"):
            self._ad_nullspace = self.operator._ad_nullspace
        self._ad_near_nullspace = None
        if hasattr(self.operator, "_ad_near_nullspace"):
            self._ad_near_nullspace = self.operator._ad_near_nullspace

    def set_operator(self, arg0):
        self.operator = arg0
        if hasattr(self.operator, "_ad_nullspace"):
            self._ad_nullspace = self.operator._ad_nullspace
        if hasattr(self.operator, "_ad_near_nullspace"):
            self._ad_near_nullspace = self.operator._ad_near_nullspace

        self.block_helper = PETScKrylovSolveBlockHelper()
        return dolfin.PETScKrylovSolver.set_operator(self, arg0)

    def set_operators(self, arg0, arg1):
        self.operator = arg0
        if hasattr(self.operator, "_ad_nullspace"):
            self._ad_nullspace = self.operator._ad_nullspace
        if hasattr(self.operator, "_ad_near_nullspace"):
            self._ad_near_nullspace = self.operator._ad_near_nullspace

        self.pc_operator = arg1
        self.block_helper = PETScKrylovSolveBlockHelper()
        return dolfin.PETScKrylovSolver.set_operators(self, arg0, arg1)

    def solve(self, *args, **kwargs):
        annotate = annotate_tape(kwargs)

        if annotate:
            if len(args) == 3:
                block_helper = PETScKrylovSolveBlockHelper()
                A = args[0]
                x = args[1]
                b = args[2]
            elif len(args) == 2:
                block_helper = self.block_helper
                A = self.operator
                x = args[0]
                b = args[1]

            u = x.function
            parameters = self.parameters.copy()
            nonzero_initial_guess = parameters["nonzero_initial_guess"] or False
            ksp_options_prefix = dolfin.PETScKrylovSolver.ksp(self).getOptionsPrefix()

            tape = get_working_tape()
            sb_kwargs = PETScKrylovSolveBlock.pop_kwargs(kwargs)
            block = PETScKrylovSolveBlock(A, x, b,
                                          krylov_solver_parameters=parameters,
                                          block_helper=block_helper,
                                          nonzero_initial_guess=nonzero_initial_guess,
                                          pc_operator=self.pc_operator,
                                          krylov_method=self.method,
                                          krylov_preconditioner=self.preconditioner,
                                          ksp_options_prefix=ksp_options_prefix,
                                          _ad_nullspace=self._ad_nullspace,
                                          _ad_near_nullspace=self._ad_near_nullspace,
                                          ad_block_tag=self.ad_block_tag,
                                          **sb_kwargs)
            tape.add_block(block)

        out = dolfin.PETScKrylovSolver.solve(self, *args, **kwargs)

        if annotate:
            block.add_output(u.create_block_variable())

        return out
