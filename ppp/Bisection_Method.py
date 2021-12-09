#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes : NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7 : Tue Nov  5 10:26:28 2019
"""

def bisection_method(f, a, b, tolerance=1E-16):
    """
    Returns the root of function f in the domain [a, b] using the bisection
    method with a tolerance of tol.

    Parameters
    ---------- 
    f : Function
        The function on which to perform the bisection method, seeking roots.
    a : Float
        The lower limit of the domain on which to perform the bisection method.
    b : Float
        The upper limit of the domain on which to perform the bisection method.
    tolerance : Float
        The tolerance of the bisection method. The default is machine-precision.
    """
    c = (a+b)/2
    while (b-a)/2 > tolerance:
        if f(c) == 0:
            return c
        elif f(a)*f(c) < 0:
            b = c
        else :
            a = c
        c = (a+b)/2

    return c