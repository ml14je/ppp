#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes : NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7 : Tue Nov  5 10:18:14 2019
"""

import numpy as np

def newton_raphson(f, init_val, sup_output=True, max_iteration=250,
                   tolerance=1E-16, eps=1E-8):
    """
    Newton--Raphson method which utilises the Secant-Method to
    approximate the derivative of function f. The result should return the
    closest root of f to init_val.

    Parameters
    ----------
    f : Function
        The function of which the Newton--Raphson method will attempt find the
        root.
    init_val : Float
        The initial guess of the Newton--Raphson method.
    sup_output : Boolean
        This will determine whether to print or not the various adaptions,
        such as whether the maximum number of iterations has been reached. The
        default is False.
    max_iteration : Integer
        Maximum number of iterations of the Newton-Raphson method. Default is
        250.
    """
    F = f(init_val)
    error, iteration = abs(F), 1
    val = init_val
    v0 = init_val


    while True:
        f_prime = (f(val+eps/2)-f(val-eps/2))/eps
        val -= F/f_prime
        F = f(val)
        error = abs(F)

        if error < tolerance:
            return val

        # print(val, v0, error)
        iteration +=1
        if v0 == 0:
            v0 += eps

        if abs((val-v0)/v0) < 1e-14:
            if not sup_output:
                print('Stationary point')

            return None

        if iteration > max_iteration:
            if not sup_output:
                print('Could not converge to selected precision within {} \
iterations, derivative: {:.2e}. \n'.format(max_iteration, abs(f_prime)))
            return None

        v0 = val

    return val

def newton_system(F, x, tolerance = 1e-10, max_iter = 200, t=complex, eps=1e-2):
    x = x.astype(t)
    F_vec = F(x)
    F_norm = norm(F_vec, F_vec)
    counter = 0
    Jacobian = lambda x : jac(F, x, t=t, eps=eps)

    while (F_norm > tolerance) and (counter < max_iter):
        J = Jacobian(x)
        delta = np.linalg.solve(J, -F_vec)
        x += delta
        F_vec = F(x)
        F_norm = norm(F_vec, F_vec)
        counter += 1

    return x, counter

def jac(F, x, eps=1e-2, t=complex):
    n = len(x)
    J = np.zeros((n, n), dtype=t)
    for i in range(n):
        vec = np.copy(x)
        vec[i] *= 1 + eps
        f1, f2 = F(vec), F(x)
        h = vec[i]-x[i]
        if h == 0:
            h = eps

        for j in range(n):
            J[j, i] = (f1[j]-f2[j])/h

    return J

def norm(v1, v2):
    assert len(v1) == len(v2)
    n = len(v1)
    S = 0
    for i in range(n):
        S += v1[i] * v2[i].conjugate()

    return np.sqrt(S/n)

def F(x, r = 28, β = 8/3, σ = 10):
    xx = np.copy(x)

    xx[0] = σ*(x[1] - x[0])
    xx[1] = x[0] * (r - x[2]) - x[1]
    xx[2] = x[0] * x[1] - β*x[2]

    return xx

def test():
    r, β, σ = 28, 8/3, 10
    lorenz = lambda x : F(x)
    xx = np.array([
        -np.sqrt(β*(r-1)),
        -np.sqrt(β*(r-1)),
        r-1
       ])

    print(xx, newton_system(lorenz, xx-244))

def test2():
    function = lambda x : np.array([np.sin(x[0]), np.cos(x[1])])
    xx = np.array([
        0, np.pi/2
       ])

    print(xx.real, newton_system(function, xx+1e-1*np.ones(2))[0].real)


if __name__ == '__main__':
    test()


