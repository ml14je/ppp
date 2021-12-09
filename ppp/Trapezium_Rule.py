#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes : NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7 : Tue Nov  5 10:31:41 2019
"""

def trapezium_rule(f, a, b, n):
    """
    Integrates the function f between a and b with n + 1 grid points using the
    trapezium rule.

    Parameters
    ---------- 
    f : Function
        The function to integrate.
    a : Float
        The lower limit of the definite integral.
    b : Float
        The upper limit of the definite integral.
    n : Integer
        The number of sections on which to perform the trapezium rule.
    """
    S, h = (f(a)+f(b))/2, (b-a)/n

    for i in range(1, n):
        S+=f(a+i*h)
    
    return h*S

def trapz(f_vals, x_vals):
    import numpy as np

    dx_vals = x_vals[1:] - x_vals[:-1]
    S = np.sum(dx_vals*(f_vals[1:]+f_vals[:-1])/2)

    return S
        

def trapezium_rule2(f_vals, x_vals=None, dx=1, ax=1):
    """
    Integrates the function f between a and b with n + 1 grid points using the
    trapezium rule.

    Parameters
    ---------- 
    f : Function
        The function to integrate.
    a : Float
        The lower limit of the definite integral.
    b : Float
        The upper limit of the definite integral.
    n : Integer
        The number of sections on which to perform the trapezium rule.
    """
    import numpy as np

    if x_vals is not None:
        I = np.trapz(f_vals, x=x_vals, axis=ax)

    else:
        I = np.trapz(f_vals, dx=dx, axis=ax)

    return I

def trapezium_rule_2D(f_vals, x_vals, y_vals):
    import numpy as np
    assert f_vals.shape[0] == len(y_vals)
    assert f_vals.shape[1] == len(x_vals)

    intermediate = np.trapz(f_vals, x_vals[:, None], axis=0)
    return np.trapz(intermediate, y_vals)
