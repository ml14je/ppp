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

    while error > tolerance:
        f_prime = (f(val+eps/2)-f(val-eps/2))/eps

        val -= F/f_prime
        F = f(val)
        error = abs(F)
        iteration +=1

        if iteration > max_iteration:
            if not sup_output:
                print('Could not converge to selected precision within {} \
iterations.\n'.format(max_iteration))
            return val
    
    return val

def N_dim_newton_raphson(F, vals, sup_output=True, max_iteration=500,
                         tolerance=1E-16, eps=1E-8):
    """
    N-dimensional Newton--Raphson method which utilises the Secant-Method to
    approximate the Jacobian matrix. The result should yield roots of F(X)=0,
    where X = vals.
    
    Parameters
    ----------
    F : Function
        Vector function of which we attempt to solve roots F(X)=0.
    vals : numpy.ndarray
        Array of initial guesses for the roots of F(X)=0.
    tol : Float
        Error tolerance for p2 norm of F(X).
    max_iteration : Integer
        Maximum number of iterations for the Newton--Raphson method.
    """
    from P_Norms import p2_norm
    from numpy.linalg import inv, LinAlgError

    error, iteration  = p2_norm(F(vals)), 1

    while error > tolerance:
        J = jacobian_matrix(F, vals, eps)

        try:
            J_inv = inv(J)
        except LinAlgError as err:
            if 'Singular matrix' in str(err):
                if not sup_output:
                    print('Singular Matrix. Return result.\n')
                return vals

        vals -= np.matmul(J_inv, F(vals))
        error = p2_norm(F(vals))

        if iteration > max_iteration:
            if not sup_output:
                print('Could not converge to machine precision within {} \
#iterations.\n'.format(max_iteration))
            return vals
        
        iteration +=1
    
    return vals

def jacobian_matrix(F, vals, eps):
    """
    Generates the Jacobian matrix of vector function F around variable values
    var.
    
    Parameters
    ----------
    F : Function
        Vector function of which one constructs the Jacobian matrix.
    vals : numpy.ndarray
        Vector values at which one constructs the Jacobian matrix.
    eps : Float
        The small change in the vector values to approximate the partial
        derivatives in the constructed Jacobian matrix. The default value is
        1E-12.
    """
    
    m = len(vals)
    M = np.zeros((m, m))
    for i in range(m):
        dvals1, dvals2 = np.copy(vals), np.copy(vals)
        dvals1[i] += eps/2
        dvals2[i] -= eps/2
        M[:, i] = (F(dvals1)-F(dvals2))/eps

    return M
