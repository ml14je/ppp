#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes : NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7 : Tue Nov  5 10:56:48 2019
"""

import numpy as np

def one_norm(M):
    """
    Returns the one-norm of n x m matrix M.
    
    Parameters
    ----------
    M : numpy.ndarray
        n x m matrix of which we return the one-norm.
    """
    n, m = M.shape
    colsum = np.zeros(m)
    
    for j in range(m):
        colsum[j] = np.sum(np.abs(M[:, j]))
        
    return np.sort(colsum)[m-1]

def two_norm(M):
    """
    Returns the two-norm of n x n matrix M.
    
    Parameters
    ----------
    M : numpy.ndarray
        n x m matrix of which we return the two-norm.
    """
    n, m = M.shape
    assert n == m #Requires M be a square matrix
    B = np.matmul(M.T, M)
    
    return np.sqrt(rho(B))

def inf_norm(M):
    """
    Returns the infinity-norm of n x m matrix M.
    
    Parameters
    ----------
    M : numpy.ndarray
        n x m matrix of which we return the infity-norm.
    """
    n, m = M.shape
    rowsum = np.zeros(n)
    
    for j in range(n):
        rowsum[j] = np.sum(np.abs(M[j, :]))
        
    return np.sort(rowsum)[n-1]

def frobenius_norm(M):
    """
    Returns the frobenius norm of matrix M.
    
    Parameters
    ----------
    M : numpy.ndarray
        n x m matrix of which we return the Frobenius norm.
    """
    n, m = M.shape
    S = 0
    
    for i in range(m):
        for j in range(n):
            S += M[j, i]**2 #S is the sum of all matrix elements squared
            
    return np.sqrt(S)

def eig_vals(M):
    """
    Returns the eigenvalues of square matrix M. This function is
    essentially a wrapper for the numpy function np.linalg.eig. Note that
    on the return line, writing instead np.linalg.eig(M)[1] would return
    the eigenvectors.
    
    Parameters
    ----------
    M : numpy.ndarray
        Square matrix of which we return the eigenvalues.
    """
    n, m = M.shape
    assert n == m #Requires M be a square matrix
    
    return np.linalg.eig(M)[0]

def rho(M):
    """
    Returns the spectral radius of square matrix M.
    
    Parameters
    ----------
    M : numpy.ndarray
        Square matrix of which we return the spectral radius.
    """
    n, m = M.shape
    assert n == m #Requires M be a square matrix
    eigvals = eig_vals(M)
    
    modlam = np.array([np.abs(eigval) for eigval in eigvals])
    
    return np.sort(modlam)[n-1]

if __name__ == '__main__':
    digit_precision = 3
    A = np.array([
            [-4, 3, 3],
            [1, -2, 1],
            [0, 1, -5]
            ])

    print('One-norm: {}'.format(round(inf_norm(A), digit_precision)))
    print('Two-norm: {}'.format(round(two_norm(A), digit_precision)))
    print('Infinity-norm: {}'.format(round(one_norm(A), digit_precision)))
    print('Frobenius norm: {}'.format(round(frobenius_norm(A),
          digit_precision)))
    print('Spectral radius: {}'.format(round(rho(A), digit_precision)))