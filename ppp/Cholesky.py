#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes : NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7 : Tue Nov  5 10:47:49 2019
"""

import numpy as np

def LU_decomposition(a, b, c):
    """
    Decomposes out tridiagonal matrix as the product of a lower-triangular
    matrix L* and upper-triangular matrix U*.
    
    Parameters
    ----------
    a : numpy.ndarray
        Lower diagonal values.
    b : numpy.ndarray
        Diagonal values.
    c : numpy.ndarray
        Upper diagonal values.
    """

    n = len(a)
    l, v, w = np.zeros(n, dtype=complex), np.zeros(n+1, dtype=complex), \
                np.zeros(n, dtype=complex)
    v[0]= b[0]
    for i in range(n):
        w[i] = c[i]
        l[i]=a[i]/v[i]
        v[i+1] = b[i+1]-l[i]*w[i]

    return l, v, w

def diag(M):
    return M.diagonal(-1), M.diagonal(0), M.diagonal(1)

def cholesky_solver(M, beta):
    """
    Returns the solution X to M.X = beta, where M is a tri-diagonal matrix.
    
    Parameters
    ----------
    M : numpy.ndarray
        Tridigonal matrix
    beta : numpy.ndarray
        Vector B in Ax = B, for which we solve for vector x.   
    """

    N = M.shape[0]
    a, b, c = diag(M)
    l, v, w = LU_decomposition(a, b, c)
    U, L = np.eye(N, dtype=complex), np.eye(N, dtype=complex)
    np.fill_diagonal(L[1:], l)
    np.fill_diagonal(U, v)
    np.fill_diagonal(U[:,1:], w)

    x, y = np.zeros(N, dtype=complex), np.zeros(N, dtype=complex)
    y[0] = beta[0]
    
    for i in range(1, N): #First solve for y in L* y = beta
        y[i] = beta[i] - l[i-1]*y[i-1]
        w[i-1] = c[i-1]
    x[-1] = (y[-1]/v[-1])
    for i in reversed(range(0, N-1)):
        x[i] = (y[i]-w[i]*x[i+1])/v[i]
      
    return x

if __name__ == '__main__':
    N = 500
    a = np.random.randint(1, 100, size=N-1)
    b = np.random.randint(1, 100, size=N)
    c = np.random.randint(1, 100, size=N-1)
    M = np.zeros((N, N), dtype = complex)
    np.fill_diagonal(M, b)
    np.fill_diagonal(M[1:], a)
    np.fill_diagonal(M[:,1:], c)
    l, v, w = LU_decomposition(a, b, c)
    U, L = np.eye(N, dtype = complex), np.eye(N, dtype = complex)
    np.fill_diagonal(L[1:], l)
    np.fill_diagonal(U, v)
    np.fill_diagonal(U[:,1:], w)
    beta = np.random.randint(10, size=N)
    print(np.real(np.round(cholesky_solver(M, np.ones(N))-\
                   np.linalg.solve(M, np.ones(N)))))