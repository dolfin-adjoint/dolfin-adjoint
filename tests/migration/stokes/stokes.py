"""
Very stupid scheme for decoupled stationary Stokes + heat equation:

Given nu and f, find (u, p) such that

  (nu grad(u), grad(v)) + (p, div(v)) = (f, v)
                          (div(u), q) = 0

for all (v, q).

Given velocity u, find T such that

  (Dt(T), s) + (s*div(v) + (grad(T), grad(s)) = (1, s)

for all s

"""

from fenics import *
from fenics_adjoint import *

from numpy.random import rand


def stokes(W, nu, f):
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    a = (nu * inner(grad(u), grad(v))
         + p * div(v) + q * div(u)) * dx
    L = inner(f, v) * dx
    return (a, L)


def temperature(X, kappa, v, t_, k):
    t = TrialFunction(X)
    s = TestFunction(X)

    F = ((t - t_) / k * s + inner(kappa * grad(t), grad(s))
         + dot(v, grad(t)) * s) * dx - s * dx
    (a, L) = system(F)
    return (a, L)


def flow_boundary_conditions(W):
    u0 = Constant((0.0, 0.0))
    bottom = DirichletBC(W.sub(0), (0.0, 0.0), "near(x[1], 0.0)")
    top = DirichletBC(W.sub(0), (0.0, 0.0), "near(x[1], 1.0)")
    left = DirichletBC(W.sub(0).sub(0), 0.0, "near(x[0], 0.0)")
    right = DirichletBC(W.sub(0).sub(0), 0.0, "near(x[0], 1.0)")
    bcs = [bottom, top, left, right]
    return bcs


def temperature_boundary_conditions(Q):
    bc = DirichletBC(Q, 0.0, "near(x[1], 1.0)")
    return [bc]


n = 16
mesh = UnitSquareMesh(n, n)
X = FunctionSpace(mesh, "CG", 1)


def main(ic, annotate=False):

    # Define meshes and function spaces
    cg2 = VectorElement("CG", triangle, 2)
    cg1 = FiniteElement("CG", triangle, 1)
    ele = MixedElement([cg2, cg1])
    W = FunctionSpace(mesh, ele)

    # Define boundary conditions
    flow_bcs = flow_boundary_conditions(W)
    temp_bcs = temperature_boundary_conditions(X)

    # Temperature variables
    T_ = ic.copy(deepcopy=True, annotate=annotate)
    T = ic.copy(deepcopy=True, annotate=annotate)

    # Flow variable(s)
    w = Function(W)
    (u, p) = split(w)

    # Some parameters
    Ra = Constant(1.e4)
    nu = Constant(1.0)
    kappa = Constant(1.0)
    timestep = 0.1

    # Define flow equation
    g = as_vector((Ra * T_, 0))
    flow_eq = stokes(W, nu, g)

    # Define temperature equation
    temp_eq = temperature(X, kappa, u, T_, timestep)

    # Time loop
    t = 0.0
    end = 1.0
    while (t <= end):

        solve(flow_eq[0] == flow_eq[1], w, flow_bcs, annotate=annotate)

        solve(temp_eq[0] == temp_eq[1], T, temp_bcs, annotate=annotate)
        T_.assign(T, annotate=annotate)
        # plot(T)

        t += timestep

    return T_


if __name__ == "__main__":

    # Run model
    T0_expr compute_derivative(] * x[1]) + 0.01 * cos(pi * x[0] / l) * sin(pi * x[1] / h)"
    T0 = Expression(T0_expr, l=1.0, h=1.0, degree=1)
    ic = interpolate(T0, X)
    ic = Function(X, ic.vector())
    T = main(ic, annotate=True)

    J = assemble(T * T * dx)
    dJdic = compute_derivative(J, Control(ic))

    h = Function(X)
    h.vector()[:] = rand(X.dim())
    dJdic = h._ad_dot(dJdic)

    def Jhat(ic):
        T = main(ic, annotate=False)
        return assemble(T * T * dx)

    minconv = taylor_test(Jhat, ic, h, dJdic)
    assert minconv > 1.9
