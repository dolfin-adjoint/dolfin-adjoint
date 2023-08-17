import dolfin

from fenics_adjoint.utils import function_from_vector


from . import SolveLinearSystemBlock
from .assembly import assemble_adjoint_value


class KrylovSolveBlockHelper(object):
    def __init__(self):
        self.forward_solver = None
        self.adjoint_solver = None

    def reset(self):
        self.forward_solver = None
        self.adjoint_solver = None


class KrylovSolveBlock(SolveLinearSystemBlock):
    def __init__(self, A, u, b,
                 krylov_solver_parameters,
                 block_helper, nonzero_initial_guess,
                 pc_operator, krylov_method,
                 krylov_preconditioner,
                 **kwargs):
        super(KrylovSolveBlock, self).__init__(A, u, b, **kwargs)

        self.krylov_solver_parameters = krylov_solver_parameters
        self.block_helper = block_helper
        self.nonzero_initial_guess = nonzero_initial_guess
        self.pc_operator = pc_operator
        self.method = krylov_method
        self.preconditioner = krylov_preconditioner

        if self.nonzero_initial_guess:
            # Here we store a variable that isn't necessarily a dependency.
            # This means that the graph does not know that we depend on this BlockVariable.
            # This could lead to unexpected behaviour in the future.
            # TODO: Consider if this is really a problem.
            self.func.block_variable.save_output()
            self.initial_guess = self.func.block_variable

        if self.pc_operator is not None:
            self.pc_operator = self.pc_operator.form
            for c in self.pc_operator.coefficients():
                self.add_dependency(c)

    def _create_initial_guess(self):
        r = super(KrylovSolveBlock, self)._create_initial_guess()
        if self.nonzero_initial_guess:
            dolfin.Function.assign(r, self.initial_guess.saved_output)
        return r

    def _assemble_and_solve_adj_eq(self, dFdu_adj_form, dJdu, compute_bdy):
        dJdu_copy = dJdu.copy()
        bcs = self._homogenize_bcs()

        solver = self.block_helper.adjoint_solver
        if solver is None:
            solver = dolfin.KrylovSolver(self.method, self.preconditioner)

            if self.assemble_system:
                rhs_bcs_form = dolfin.inner(dolfin.Function(self.function_space),
                                            dFdu_adj_form.arguments()[0]) * dolfin.dx
                A, _ = dolfin.assemble_system(dFdu_adj_form, rhs_bcs_form, bcs)

                if self.pc_operator is not None:
                    P = self._replace_form(self.pc_operator)
                    P, _ = dolfin.assemble_system(P, rhs_bcs_form, bcs)
                    solver.set_operators(A, P)
                else:
                    solver.set_operator(A)
            else:
                A = assemble_adjoint_value(dFdu_adj_form)
                [bc.apply(A) for bc in bcs]

                if self.pc_operator is not None:
                    P = self._replace_form(self.pc_operator)
                    P = assemble_adjoint_value(P)
                    [bc.apply(P) for bc in bcs]
                    solver.set_operators(A, P)
                else:
                    solver.set_operator(A)

            self.block_helper.adjoint_solver = solver

        solver.parameters.update(self.krylov_solver_parameters)
        [bc.apply(dJdu) for bc in bcs]

        adj_sol = dolfin.Function(self.function_space)
        solver.solve(adj_sol.vector(), dJdu)

        adj_sol_bdy = None
        if compute_bdy:
            adj_sol_bdy = function_from_vector(self.function_space, dJdu_copy - assemble_adjoint_value(
                dolfin.action(dFdu_adj_form, adj_sol)))

        return adj_sol, adj_sol_bdy

    def _forward_solve(self, lhs, rhs, func, bcs, **kwargs):
        solver = self.block_helper.forward_solver
        if solver is None:
            solver = dolfin.KrylovSolver(self.method, self.preconditioner)
            if self.assemble_system:
                A, _ = dolfin.assemble_system(lhs, rhs, bcs)
                if self.pc_operator is not None:
                    P = self._replace_form(self.pc_operator)
                    P, _ = dolfin.assemble_system(P, rhs, bcs)
                    solver.set_operators(A, P)
                else:
                    solver.set_operator(A)
            else:
                A = assemble_adjoint_value(lhs)
                [bc.apply(A) for bc in bcs]
                if self.pc_operator is not None:
                    P = self._replace_form(self.pc_operator)
                    P = assemble_adjoint_value(P)
                    [bc.apply(P) for bc in bcs]
                    solver.set_operators(A, P)
                else:
                    solver.set_operator(A)
            self.block_helper.forward_solver = solver

        if self.assemble_system:
            system_assembler = dolfin.SystemAssembler(lhs, rhs, bcs)
            b = dolfin.Function(self.function_space).vector()
            system_assembler.assemble(b)
        else:
            b = assemble_adjoint_value(rhs)
            [bc.apply(b) for bc in bcs]

        solver.parameters.update(self.krylov_solver_parameters)
        solver.solve(func.vector(), b)
        return func
