#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes : NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7 : Tue Nov  5 10:51:15 2019
"""

import numpy as np

def diag_rescale(M):
    """
    Rescales each row to ensure the row's diagonal element is 1.
    
    Parameters
    ----------
    M : numpy.ndarray
        Square matrix of which we return the diagonal re-scaling.
    """
    n, m = M.shape
    assert n == m #We require M to be a square matrix
    M_new = np.copy(M)
    for i in range(n):
        Mii = M[i, i]
        M_new[i, :] /= Mii 
    
    return M_new

def L_decomp(M):
    """
    Decomposes square matrix M and returns its lower triagonal.
    
    Parameters
    ----------
    M : numpy.ndarray
        Square matrix of which we return the L-decomposition.
    """
    n, m = M.shape
    assert n == m #We require M to be a square matrix
    L = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if j<i:
                L[i, j] = M[i, j]
                
    return -L

def U_decomp(M):
    """
    Decomposes square matrix M and returns its upper triagonal.
    
    Parameters
    ----------
    M : numpy.ndarray
        Square matrix of which we return the U-decomposition.
    """
    n, m = M.shape
    assert n == m #We require M to be a square matrix
    U = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if j>i:
                U[i, j] = M[i, j]
                
    return -U

def Jacobi(M):
    """
    Returns the Jacobi matrix of square matrix M.
    
    Parameters
    ----------
    M : numpy.ndarray
        Square matrix of which we return the Jacobi matrix.
    """
    B = diag_rescale(M)
    L, U = L_decomp(B), U_decomp(B)
    
    return L + U

def Gauss_Seidel(M):
    """
    Returns the Guass-Seidel matrix of square matrix M.
    
    Parameters
    ----------
    M : numpy.ndarray
        Square matrix of which we return the Gauss-Seidel matrix.
    """
    from numpy.linalg import inv
    n, m = M.shape
    assert n == m #We require M to be a square matrix
    B, I = diag_rescale(M), np.eye(n) #np.eye(n) returns I_n
    L, U = L_decomp(B), U_decomp(B)

    return np.matmul(inv(I-L), U)

def SOR(M, ω):
    """
    Returns the SOR matrix of square matrix M with SOR parameter ω.
    
    Parameters
    ----------
    M : numpy.ndarray
        Square matrix of which we return the SOR matrix.
    ω : Float
        SOR parameter which typically lies in the domain [1, 2].
    """
    from numpy.linalg import inv
    n, m = M.shape
    assert n == m #We require M to be a square matrix
    B, I = diag_rescale(M), np.eye(n) #np.eye(n) returns I_n
    L, U = L_decomp(B), U_decomp(B)
    
    return np.matmul(inv(I-ω*L), (1-ω)*I+ω*U)