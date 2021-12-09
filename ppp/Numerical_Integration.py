#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Wed Aug 19 17:42:20 2020
"""
import numpy as np
from scipy.sparse import csr_matrix as sp

def trapezium_matrix(x):
    """
    This function will return the numerical integration matrix for the
    trapezium rule. Multiplying the return matrix to a functional array will
    give the numerical integration from x[0] to x[1:].

    Parameters
    ----------
    x : numpy array
        The domain over which to perform the numerical integration

    Returns
    -------
    scipy.sparse matrix
        The sparse matrix for the numerical trapezium rule.
    """
    N = len(x) - 1
    T = np.tril(np.ones((N, N)))
    dx = np.diag(np.diff(x))

    weights = (np.eye(N+1, k=1)+np.eye(N+1))[:-1, :]

    return sp(T @ dx/2 @ weights)

if __name__ == '__main__':
    N = 500
    # x = np.linspace(0, np.pi, N+1)
    x = np.concatenate((np.linspace(0, np.pi/4, N), np.linspace(np.pi/4, np.pi, N)[1:]))
    y = np.sin(x)

    T = np.tril(np.ones((N, N)))
    M = trapezium_matrix(x)
