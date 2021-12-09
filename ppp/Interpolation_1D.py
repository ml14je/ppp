#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes : NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7 : Tue Nov  5 12:16:36 2019
"""

import numpy as np

def series(coeff_vec, x):
    """
    Returns the power series evaluation of x with integer powers indicated by
    the coefficients in coeff_vec.

    Parameters
    ---------- 
    coeff_vec : numpy.ndarray
        Coefficients of powers series.
    x : numpy.ndarray
        Values at which to evaluate the power series.
    """
    S = 0
    for i, a in enumerate(coeff_vec):
        S+=a*(x**i)
    
    return S

def series_der(coeff_vec, x):
    
    S = 0
    for i, a in enumerate(coeff_vec):
        if i != 0:
            S+=i*a*(x**(i-1))

    return S

def lagrange_interp(x_arr, X, U):
    """
    Let X = {x0, x1, ..., xn} be n+1 distinct real numbers (x0 < x1 <...< xn)
    with associated function values U = {u0, u1, ..., un}. Returns the
    Lagrangian polynomial that interpolates {X, U} on x_arr.

    Parameters
    ----------
    x_arr : numpy.ndarray
        The array of points on which to interpolate onto.
    X : numpy.ndarray
        The known points from which to interpolate.
    U : numpy.ndarray
        The associated function values of X.
    """

    assert len(X) == len(U) #both X and U must be of the same length
    n, S = len(X), 0
    for i in range(n):
        P = 1
        for j in range(n):
            if j!=i:
                P*=(x_arr-X[j])/(X[i]-X[j])
                
        S+=P*U[i]
    
    return S

def divided_diff(x_arr, X, U):
    """
    Let X = {x0, x1, ..., xn} be n+1 distinct real numbers (x0 < x1 <...< xn)
    with associated function values U = {u0, u1, ..., un}. Returns the
    nth-degree divided-difference polynomial that interpolates {X, U} on x_arr.
    
    Parameters
    ----------
    x_arr : numpy.ndarray
        The array of points on which to interpolate onto.
    X : numpy.ndarray
        The known points from which to interpolate.
    U : numpy.ndarray
        The associated function values of X.
    """

    assert len(X) == len(U) #both X and U must be of the same length
    n = len(X)
    D, B = np.zeros(n), np.copy(U)

    for i in range(1, n):
        A = np.copy(B)
        D[i-1] = B[i-1]
        for j in range(i, n):
            B[j]=(A[j]-A[j-1])/(X[j]-X[j-i])

    D[-1] = B[n-1]
    print('\nDivided differences of {X, U} on [a, b]:')
    for i in range(len(D)):
        print('\tD_{} = {}'.format(i, round(D[i], 5)))

    S, P = 0, 1
    for i in range(n):
        S+=P*D[i]
        P*=(x_arr-X[i])

    return S

def minimax_func(f, f_prime, var, a, b):
    """
    Returns the vector function necessary to equal zero when finding the
    minimax polynomial.
    
    Parameters
    ----------
    f : Function
        The function of which we are attempting to construct a minimax
        interpolation polynomial.
    f_prime : Function
        The derivative of the function of which we are attempting to construct
        a minimax interpolation polynomial.
    var : numpy.ndarray
        The variable array of values of which we evaluate the vector function.
        In theory, for our minimax approximation, we should expect these, if
        correct, to be roots of the vector function.
    a : Float
        The lower limit of the domain over which we construct the minimax
        polynomial.
    b : Float
        The upper limit of the domain over which we construct the minimax
        polynomial.
    """

    n, F = int(len(var)/2-1),  np.zeros(len(var))
    for i in range(n):
        F[i]=f(var[i])-series(var[n:-1], var[i])+((-1)**i)*var[-1]
        F[n+i]=f_prime(var[i])-series_der(var[n:-1], var[i])
    F[-2] = f(a)-series(var[n:-1], a)-var[-1]
    F[-1] = f(b)-series(var[n:-1], b)+((-1)**(n))*var[-1]
    
    return F

def minimax(f, a, b, n, sup_output=False, eps=1E-9):
    """
    Solves minimax system on function f(x), and
    returns parameter values [x_1, ..., x_n, a_0, ..., a_N, rho], where x_i is
    the minimax point, a_i is the coefficient to x^i in the minimax
    interpolation polynomial q_n(x) and rho is the absolute error of q_n on
    points {a, x_1, x_2, ..., x_n, b}.
    
    Parameters
    ----------    
    f : Function
        Function to which the minimax polynomial is constructed.
    a : Float
        The lower limit of the domain over which we construct the minimax
        polynomial.
    b : Float
        The upper limit of the domain over which we construct the minimax
        polynomial.
    n : Integer
        Indicates the order of the minimax interpolating polynomial.
    sup_output : Boolean
        If True, the function will supress printing the output.
    """
    from sympy.abc import x

    if __name__ == '__main__':
        from Newton_Raphson import N_dim_newton_raphson
    else:
        from PPP.Newton_Raphson import N_dim_newton_raphson
    

    var_init = np.zeros(2*(n+1))
    var_init[:n] = np.linspace(a, b, n+2)[1:-1]

    if callable(f):
        f_prime = lambda x : (f(x+eps)-f(x))/eps
        
    else:
        from sympy import lambdify, diff
        f_prime = lambdify(x, diff(f, x), "numpy")
        f = lambdify(x, f, "numpy")

    F = lambda init_vals : minimax_func(f, f_prime, init_vals, a, b)
    var = N_dim_newton_raphson(F, var_init, eps = eps, sup_output=False)

    if not sup_output:
        print('Minimax points:')
        for i in range(n):
            print('\tx_{} = {}'.format(i+1, round(var[i], 4)))
        func = 0
        for i in range(n+1):
            func += round(var[n+i], 4)*(x**i)
        print('\nMinimax polynomial:\n\tq_{}(x) = {}'.format(n, func))
        print('\nMaximum absolute error on [a, b]:\n\trho\
= {}\n\n'. format(round(var[-1], 4)))

    return lambda y: series(var[n:2*n+1], y)

if __name__ == '__main__':
    import matplotlib.pyplot as pt

    scale = .5
    from sympy.abc import x
    from sympy import exp, lambdify
    f = exp(x)
    f_approx = minimax(f, -1, 1, 3, sup_output=False, eps=1E-15)
    
    x_vals = np.linspace(-1, 1, 100)

    from Plots import plot_setup
    fig, ax = plot_setup('$x$', scale=scale)
    if not callable(f):
        f = lambdify(x, f, "numpy")

    ax.plot(x_vals, f(x_vals), label='$f(x)$')
    ax.plot(x_vals, f_approx(x_vals), label='$q_2(x)$')
    ax.legend(fontsize=scale*16)
    pt.show()
