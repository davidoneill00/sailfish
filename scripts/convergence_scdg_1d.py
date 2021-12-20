#!/usr/bin/env python3

"""
Measure the L1 error with respect to the analytic solution by integrating L1 =
1/V Integral(abs(u_computed-u_analytic)) dV over the cell using Gaussian
quadrature with order l1_order. Schaal+15 recommend using 2 orders higher
quadrature than the simulation order: l1_order = simulation_order + 2 to make
sure the measured error is that due to the simulation itself, and not from the
analysis.
"""

import argparse
import pickle
import sys
from numpy.polynomial.legendre import leggauss, Legendre
from numpy import array, pi, sin

sys.path.insert(1, ".")


def analytic(t, x):
    """
    Analytic solution from initial condition from Advection setup in
    sailfish/setups/simple1d.py
    """
    a = 0.1
    k = 2.0 * pi
    wavespeed = 1.0
    return 1.0 + a * sin(k * (x - wavespeed * t))


def leg(x, n):
    """
    Legendre polynomials scaled by sqrt(2n + 1) used as basis functions
    """
    c = [(2 * n + 1) ** 0.5 if i is n else 0.0 for i in range(n + 1)]
    return Legendre(c)(x)


def dot(u, p):
    return sum(u[i] * p[i] for i in range(u.shape[0]))


def compute_error(state):
    scheme_order = state["solver_options"]["order"]

    # Schaal+15 recommends computing the L1 error using order p + 2 quadrature,
    # where p is the order at which the simulation was run.
    l1_order = scheme_order + 2

    # Gaussian quadrature points inside cell
    gauss_points, weights = leggauss(l1_order)

    # Value of basis functions at the quadrature points
    phi_value = array([[leg(x, n) for n in range(l1_order)] for x in gauss_points])

    time = state["time"]
    mesh = state["mesh"]
    xc = mesh.zone_centers(time)
    dx = mesh.dx
    uw = state["solution"]
    num_zones = mesh.shape[0]
    num_points = len(gauss_points)
    l1 = 0.0

    for i in range(num_zones):
        for j in range(num_points):
            xsi = gauss_points[j]
            xj = xc[i] + xsi * 0.5 * dx
            u_analytic = analytic(time, xj)
            u_computed = dot(uw[i], phi_value[j])
            l1 += abs(u_computed - u_analytic) * weights[j]

    return l1[0] * dx


def main(args):
    from sailfish.driver import run
    from matplotlib import pyplot as plt

    errors = []
    resolutions = [10, 20, 40, 80, 160, 320]
    solver_options = dict(order=2, integrator="rk2")

    print(f"solver_options = {solver_options}")

    for res in resolutions:
        state = run(
            "advection",
            end_time=0.1,
            cfl_number=0.05,
            resolution=res,
            solver_options=solver_options,
        )
        err = compute_error(state)
        errors.append(err)
        print(f"run with res = {res} error = {err:.3e}")

    expected = (
        errors[0] * (array(resolutions) / resolutions[0]) ** -solver_options["order"]
    )
    plt.loglog(resolutions, errors, "-o", mfc="none", label=r"$L_1$")
    plt.loglog(resolutions, expected, label=r"$N^{-1}$")
    plt.xlabel(r"$N$")
    plt.ylabel(r"$L_1$")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    main(args.parse_args())
