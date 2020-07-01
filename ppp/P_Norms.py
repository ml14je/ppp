#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes : NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7 : Tue Nov  5 10:25:05 2019
"""

def p1_norm(v):
    """
    Evaluates the p-1 norm (also known as Manhattan Distance or Taxicab norm)
    of vector v.

    Parameters
    ---------- 
    v : numpy.ndarray
        The vector on which to perform the p-2 norm.
    """
    import numpy as np
    return np.sum(np.abs(v))

def p2_norm(v):
    """
    Evaluates the p-2 norm (also known as the Euclidean norm) of vector v.

    Parameters
    ---------- 
    v : numpy.ndarray
        The vector on which to perform the p-2 norm.
    """
    import numpy as np
    return np.sqrt(np.matmul(v, v))

def pInf_norm(v):
    """
    Evaluates the p-infinity norm of vector v.
    
    Parameters
    ---------- 
    v : numpy.ndarray
        The vector on which to perform the p-infinity norm.
    """
    import numpy as np
    return np.max(np.abs(v))
