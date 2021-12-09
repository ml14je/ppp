#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes : NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7 : Mon Nov  4 15:54:34 2019
"""

import numpy as np

def chebd(N):
    """
    Returns the chebyshev grid x of length N+1, along with the Chebyshev
    differential matrix, D, whose size is (N+1) x (N+1).

    Parameters
    ----------
    N : Integer
    """

    if N == 0:
        D, x = 0, 1
    else:
        x = np.cos(np.pi*np.linspace(0, N, N+1)/N).T[::-1]
        c = (np.array([2]+(N-1)*[1]+[2])*((-1)**\
                      np.arange(0, N+1, 1)).T)[np.newaxis,:]

        X = np.tile(x, (N+1, 1))
        dX = X-X.T
        D = (-np.matmul(c.T, (1/c)))/(dX+np.eye(N+1))
        D -= np.diag(np.sum(D.T, axis=0))

    return D, x

def int_modes(N):
    return np.cos(((2*np.arange(1, N+1, 1)-1)/(2*N))*np.pi)


def t_transform(x_val, a, b):
    """
    Returns the linear mapping of x_val from [a, b] to [-1, 1].

    Parameters
    ----------
    x_val : numpy.ndarray
        The values on which to perform the mapping.
    a : Float
        The lower limit of the domain on which to perform the linear mapping.
    b : Float
        The upper limit of the domain on which to perform the linear mapping.
    """
    return (2*x_val-(b+a))/(b-a)

def t_inv(t_val, a, b):
    """
    Returns the inverse linear mapping of x_val from [a, b] to [-1, 1].

    Parameters
    ----------
    t_val : numpy.ndarray
        The values on which to perform the mapping.
    a : Float
        The lower limit of the domain on which to perform the linear mapping.
    b : Float
        The upper limit of the domain on which to perform the linear mapping.
    """
    return ((b-a)*t_val+(b+a))/2

if __name__ == '__main__':
    from numpy.linalg import matrix_power
    from Plots import plot_setup
    import matplotlib.pyplot as pt
    N = 100
    D, t = chebd(N)
    t2 = int_modes(N)
    a, b = 0, 2
    D*= 2/(b-a) ;
    D2 = D @ D
    x = t_inv(t, a, b)
    f = np.sin(x)
    g = lambda x : np.sin(x)
    pt.plot(x, D2 @ g(x)[:, None])
    pt.plot(x, -np.sin(x))
    pt.show()