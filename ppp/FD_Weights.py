#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes : NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7 : Wed Oct 30 16:41:16 2019
"""

import numpy as np

def weights(xi, x, m):
    """
    Returns the weights c, with element c[j, i, k] containing the weight to be
    applied at xi when the k:th derivative is approximated by a stencil
    extending over x.

    Parameters
    ----------
    xi : Float
        Point at which the approximations are made
    x : np.array
        Grid points over which stencil is constructed.
    m : Integer
        Highest order derivative to be approximated.
    """
    n=len(x)-1
    c1, c4=1, x[0]-xi

    c=np.zeros((n+1,m+1))
    c[0, 0]=1;

    for i in range(1, n+1):
        mn=min(i, m)
        c2, c5 = 1, c4
        c4=x[i]-xi
        for j in range(i):
            c3=x[i]-x[j]
            c2*=c3
            if j == i-1:
                for k in reversed(range(1, mn+1)):
                    c[i, k] = c1*(k*c[i-1, k-1]-c5*c[i-1, k])/c2
                c[i, 0] = -c1*c5*c[i-1, 0]/c2
            for k in reversed(range(1, mn+1)):
                c[j, k] = (c4*c[j, k]-k*c[j, k-1])/c3
            c[j, 0] = c4*c[j, 0]/c3
        c1=c2

    return c

def fin_diff_scheme(grid, d):
    """
    Calculates the weight coefficients for a particular numpy integer grid of
    the form [x_{i-n}, ..., x_i, ..., x_{i+m}] corresponding to the input of
    np.array([-n, -n+1, ..., 0, ..., m-1, m]) for the d-order derivative. As
    well as the coefficients of the grid points, the function also returns the
    coefficient of the next order term, for which it will be a product of the
    next derivative evaluated at x_i.

    Parameters
    ----------
    grid : numpy.ndarray
        Indicates the grid of points we use to calculate the weights of the
        finite-difference.
    d : Integer
        Order of the derivative we approximate.
    """
    from math import factorial

    N = len(grid)
    if not N>d: #ensures that the number of grid points is greater than the
                #order of the derivate.
        print("Grid size (N) must be greater than the order of the derivative \
(d) being approximated. I.e, we require N > d.")
        return

    M = np.zeros((N, N))
    if type(grid) in (tuple, list, range):
        grid = np.array(grid)

    for i in range(N):
        M[i] = grid**i

    v = np.zeros(N)
    v[d] = factorial(d)

    weights = np.matmul(np.linalg.inv(M), v)
    err = np.matmul(grid**N, weights)/factorial(N)

    return weights, err

def der_calc(f, x0, dx, grid, d):
    """
    Returns the evaluation of the d-th derivative approximation at x=x0 using
    the stencil, 'grid', with the various h values, dx. Note that 'grid' is a
    1D array of integers, while dx is a 1D arrray of floats.

    Parameters
    ----------
    f : Function
        Function of which we are approximating the d-order derivative.
    x0 : Float
        The value for which we are evaluating the d-order derivative of f(x).
    dx : Float
        Indicates the uniform grid spacing between each finite-difference
        interval x_{i+1}-x_{i}.
    grid : numpy.ndarray
        Indicates the grid of points we use to calculate the weights of the
        finite-difference.
    d : Integer
        Order of the derivative we approximate.
    """
    coeffs, error = fin_diff_scheme(grid, d)
    grid = grid.reshape(len(grid), 1)
    x_vals = x0 + np.matmul(dx, grid.T)
    f_vals = f(x_vals)

    S = 0
    for i in range(len(grid)):
        S += coeffs[i]*f_vals[:, i]

    for i in range(len(dx)):
        S[i] /= dx[i]**d

    return S

if __name__ == '__main__':
    c1 = weights(0, [-2, -1, 0, 1, 2], 2).T[1]
    c2 = weights(0, [-1, 0, 1], 1).T[1]
    c3 = weights(0, [-2, -1, 0], 1).T[1]
    c1 = weights(0, [-1.5, -.5], 0)
    x = [0, 1, 2, 3, 4]
    der = 1
    for i in range(len(x)):
        print("\nFD weights for derivative {} at position {}:".format(der, i),
              weights(x[i], x, 1).T[der])


    import numpy as np
    import matplotlib.pyplot as pt
    N = 50
    dx = 1/N
    x = np.linspace(-np.pi, np.pi, N)
    D2_prime = -np.eye(N, k =-2) + 16*np.eye(N, k=-1) -30*np.eye(N) + 16*np.eye(N, k =1) - np.eye(N, k= 2)
    D2_prime[[0,1, -1, -2]] = 0
    D2_prime /= (12*dx**2)
    D2_prime[0, :5] = [21, -92, 102, -36, 5]
    D2_prime[0, :5] /= (60*dx**2)
    D2_prime[1, :4] = [176, -357, 192, -11]
    D2_prime[1, :4] /= (150*dx**2)
    D2_prime[-1, -5:] = [5,-36, 102, -92, 21]
    D2_prime[-1, -5:] /= (60*dx**2)
    D2_prime[-2, -4:] = [-11, 192, -357, 176]
    D2_prime[-2, -4:] /= (150*dx**2)

    from scipy.linalg import eig
    vals, vecs = eig(D2_prime)

    for val, vec in zip(vals, vecs.T):
        print(val.real)
        pt.plot(x, vec)
        pt.show()


