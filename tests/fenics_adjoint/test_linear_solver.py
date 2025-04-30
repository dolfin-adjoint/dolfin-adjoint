from fenics import *
from fenics_adjoint import *
from numpy.random import rand
import pytest


def test_lu_solver():
    mesh = UnitCubeMesh(4, 4, 4)

    # Define function spaces
    cg2 = VectorElement("CG", tetrahedron, 2)
    cg1 = FiniteElement("CG", tetrahedron, 1)
    ele = MixedElement([cg2, cg1])
    W = FunctionSpace(mesh, ele)

    # Boundaries
    def right(x, on_boundary):
        return x[0] > (1.0 - DOLFIN_EPS)

    def left(x, on_boundary):
        return x[0] < DOLFIN_EPS

    def top_bottom(x, on_boundary):
        return x[1] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS

    # No-slip boundary condition for velocity
    noslip = Constant((0.0, 0.0, 0.0))
    bc0 = DirichletBC(W.sub(0), noslip, top_bottom)

    # Inflow boundary condition for velocity
    inflow = Expression(("-sin(x[1]*pi)", "0.0", "0.0"), degree=2)
    bc1 = DirichletBC(W.sub(0), inflow, right)

    # Boundary condition for pressure at outflow
    zero = Constant(0)
    bc2 = DirichletBC(W.sub(1), zero, left)

    # Collect boundary conditions
    bcs = [bc0, bc1, bc2]

    # Define variational problem
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    f = Constant((1.0, 1.0, 1.0))
    a = inner(grad(u), grad(v)) * dx + div(v) * p * dx + q * div(u) * dx
    L = inner(f, v) * dx

    # Assemble system
    A, bb = assemble_system(a, L, bcs)

    # Create LU solver
    solver = LUSolver()

    # Associate operator matrix (A)
    solver.set_operator(A)

    # Solve
    U = Function(W)
    U.vector()[:] = 0.0
    solver.solve(U.vector(), bb)

    J = assemble(inner(U, U) * inner(U, U) * dx)
    Jhat = ReducedFunctional(J, Control(f))
    assert J == Jhat(f)

    h = Constant((1.0, 1.0, 1.0))
    dJdm = h._ad_dot(Jhat.derivative())
    Hm = h._ad_dot(Jhat.hessian(h))
    assert taylor_test(Jhat, f, h, dJdm=dJdm, Hm=Hm) > 2.9


@pytest.mark.skipif(
    "amg" not in krylov_solver_preconditioners(),
    reason="AMG Preconditioner not available.",
)
def test_krylov_solver_preconditioner():
    mesh = UnitCubeMesh(4, 4, 4)

    # Define function spaces
    cg2 = VectorElement("CG", tetrahedron, 2)
    cg1 = FiniteElement("CG", tetrahedron, 1)
    ele = MixedElement([cg2, cg1])
    W = FunctionSpace(mesh, ele)

    # Boundaries
    def right(x, on_boundary):
        return x[0] > (1.0 - DOLFIN_EPS)

    def left(x, on_boundary):
        return x[0] < DOLFIN_EPS

    def top_bottom(x, on_boundary):
        return x[1] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS

    # No-slip boundary condition for velocity
    noslip = Constant((0.0, 0.0, 0.0))
    bc0 = DirichletBC(W.sub(0), noslip, top_bottom)

    # Inflow boundary condition for velocity
    inflow = Expression(("-sin(x[1]*pi)", "0.0", "0.0"), degree=2)
    bc1 = DirichletBC(W.sub(0), inflow, right)

    # Boundary condition for pressure at outflow
    zero = Constant(0)
    bc2 = DirichletBC(W.sub(1), zero, left)

    # Collect boundary conditions
    bcs = [bc0, bc1, bc2]

    # Define variational problem
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    f = Constant((0.0, 0.0, 0.0))
    a = inner(grad(u), grad(v)) * dx + div(v) * p * dx + q * div(u) * dx
    L = inner(f, v) * dx

    # Form for use in constructing preconditioner matrix
    b = inner(grad(u), grad(v)) * dx + p * q * dx

    # Assemble system
    A, bb = assemble_system(a, L, bcs)

    # Assemble preconditioner system
    P, btmp = assemble_system(b, L, bcs)

    # Create Krylov solver and AMG preconditioner
    solver = KrylovSolver("tfqmr", "amg")

    # Associate operator (A) and preconditioner matrix (P)
    solver.set_operators(A, P)

    # Solve
    U = Function(W)
    U.vector()[:] = 1.0
    rtol = 1.0e-14
    atol = 1.0e-12
    solver.parameters["relative_tolerance"] = rtol
    solver.parameters["absolute_tolerance"] = atol
    solver.parameters["nonzero_initial_guess"] = False
    solver.solve(U.vector(), bb)

    J = assemble(inner(U, U) * inner(U, U) * dx)
    Jhat = ReducedFunctional(J, Control(f))
    assert np.isclose(J, Jhat(f), atol=atol, rtol=rtol)

    h = Constant((1.0, 1.0, 1.0))
    dJdm = h._ad_dot(Jhat.derivative())
    Hm = h._ad_dot(Jhat.hessian(h))
    assert taylor_test(Jhat, f, h, dJdm=dJdm, Hm=Hm) > 2.9


def test_lu_solver_function_ctrl():
    mesh = UnitCubeMesh(4, 4, 4)

    # Define function spaces
    cg2 = VectorElement("CG", tetrahedron, 2)
    cg1 = FiniteElement("CG", tetrahedron, 1)
    ele = MixedElement([cg2, cg1])
    W = FunctionSpace(mesh, ele)

    # Boundaries
    def right(x, on_boundary):
        return x[0] > (1.0 - DOLFIN_EPS)

    def left(x, on_boundary):
        return x[0] < DOLFIN_EPS

    def top_bottom(x, on_boundary):
        return x[1] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS

    # No-slip boundary condition for velocity
    noslip = Constant((0.0, 0.0, 0.0))
    bc0 = DirichletBC(W.sub(0), noslip, top_bottom)

    # Inflow boundary condition for velocity
    inflow = Expression(("-sin(x[1]*pi)", "0.0", "0.0"), degree=2)
    bc1 = DirichletBC(W.sub(0), inflow, right)

    # Boundary condition for pressure at outflow
    zero = Constant(0)
    bc2 = DirichletBC(W.sub(1), zero, left)

    # Collect boundary conditions
    bcs = [bc0, bc1, bc2]

    # Define variational problem
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    f = Function(W.sub(0).collapse())
    f.vector()[:] = 1
    a = inner(grad(u), grad(v)) * dx + div(v) * p * dx + q * div(u) * dx
    L = inner(f, v) * dx

    # Assemble system
    A, bb = assemble_system(a, L, bcs)

    # Create LU solver
    solver = LUSolver()

    # Associate operator matrix (A)
    solver.set_operator(A)

    # Solve
    U = Function(W)
    U.vector()[:] = 1.0
    solver.solve(U.vector(), bb)

    J = assemble(inner(U, U) * inner(U, U) * dx)
    Jhat = ReducedFunctional(J, Control(f))
    assert J == Jhat(f)

    h = Function(f.function_space())
    h.vector()[:] = rand(f.function_space().dim())
    dJdm = h._ad_dot(Jhat.derivative())
    Hm = h._ad_dot(Jhat.hessian(h))
    assert taylor_test(Jhat, f, h, dJdm=dJdm, Hm=Hm) > 2.9


@pytest.mark.skipif(
    "amg" not in krylov_solver_preconditioners(),
    reason="AMG Preconditioner not available.",
)
def test_krylov_solver_preconditioner_function_ctrl():
    mesh = UnitCubeMesh(4, 4, 4)

    # Define function spaces
    cg2 = VectorElement("CG", tetrahedron, 2)
    cg1 = FiniteElement("CG", tetrahedron, 1)
    ele = MixedElement([cg2, cg1])
    W = FunctionSpace(mesh, ele)

    # Boundaries
    def right(x, on_boundary):
        return x[0] > (1.0 - DOLFIN_EPS)

    def left(x, on_boundary):
        return x[0] < DOLFIN_EPS

    def top_bottom(x, on_boundary):
        return x[1] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS

    # No-slip boundary condition for velocity
    noslip = Constant((0.0, 0.0, 0.0))
    bc0 = DirichletBC(W.sub(0), noslip, top_bottom)

    # Inflow boundary condition for velocity
    inflow = Expression(("-sin(x[1]*pi)", "0.0", "0.0"), degree=2)
    bc1 = DirichletBC(W.sub(0), inflow, right)

    # Boundary condition for pressure at outflow
    zero = Constant(0)
    bc2 = DirichletBC(W.sub(1), zero, left)

    # Collect boundary conditions
    bcs = [bc0, bc1, bc2]

    # Define variational problem
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    f = Function(W.sub(0).collapse())
    f.vector()[:] = 1
    a = inner(grad(u), grad(v)) * dx + div(v) * p * dx + q * div(u) * dx
    L = inner(f, v) * dx

    # Form for use in constructing preconditioner matrix
    b = inner(grad(u), grad(v)) * dx + p * q * dx

    # Assemble system
    A, bb = assemble_system(a, L, bcs)

    # Assemble preconditioner system
    P, btmp = assemble_system(b, L, bcs)

    # Create Krylov solver and AMG preconditioner
    solver = KrylovSolver("tfqmr", "amg")

    # Associate operator (A) and preconditioner matrix (P)
    solver.set_operators(A, P)

    # Solve
    U = Function(W)
    U.vector()[:] = 1.0
    rtol = 1.0e-14
    atol = 1.0e-12
    solver.parameters["relative_tolerance"] = rtol
    solver.parameters["absolute_tolerance"] = atol
    solver.parameters["nonzero_initial_guess"] = True
    solver.solve(U.vector(), bb)
    J = assemble(inner(U, U) * inner(U, U) * dx)
    Jhat = ReducedFunctional(J, Control(f))
    assert np.isclose(J, Jhat(f), atol=atol, rtol=rtol)

    h = Function(f.function_space())
    h.vector()[:] = rand(f.function_space().dim())
    dJdm = h._ad_dot(Jhat.derivative())
    Hm = h._ad_dot(Jhat.hessian(h))
    assert taylor_test(Jhat, f, h, dJdm=dJdm, Hm=Hm) > 2.9


class top_half(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] > 0.5


class top_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return abs(1 - x[1]) < 1e-10


def test_LU_solver_ident_zeros():
    """
    Test using ident zeros to restrict half of the domain
    """
    mesh = UnitSquareMesh(10, 10)
    cf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    top_half().mark(cf, 1)

    ff = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    top_boundary().mark(ff, 1)

    dx = Measure("dx", domain=mesh, subdomain_data=cf)

    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(grad(u), grad(v)) * dx(1)
    w = Function(V)
    with stop_annotating():
        w.assign(project(Expression("x[0]", degree=1), V))
    rhs = w**3 * v * dx(1)
    A = assemble(a, keep_diagonal=True)
    A.ident_zeros()
    b = assemble(rhs)
    bc = DirichletBC(V, Constant(1), ff, 1)
    bc.apply(A, b)
    solver = LUSolver("mumps")
    uh = Function(V)
    solver.solve(A, uh.vector(), b)

    J = assemble(inner(uh, uh) * dx(1))

    Jhat = ReducedFunctional(J, Control(w))
    with stop_annotating():
        w1 = project(Expression("x[0]*x[1]", degree=2), V)
    results = taylor_to_dict(Jhat, w, w1)
    assert min(results["R0"]["Rate"]) > 0.95
    assert min(results["R1"]["Rate"]) > 1.95
    assert min(results["R2"]["Rate"]) > 2.95
