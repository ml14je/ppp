#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes : NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7 : Wed Nov  6 09:51:22 2019
"""

import numpy as np

def A_matrix(I, A0, A1):
    O = 0*I
    return np.block([
        [A1, A0],
        [-I, O]
        ])  

def B_matrix(I, A2):
    O = 0*I
    return np.block([
        [A2, O],
        [O, I]
        ])

def quadratic_evp(I, A0, A1, A2):
    from scipy.linalg import eig
    import numpy as np

    A = A_matrix(I, A0, A1) ; B = B_matrix(I, A2)
    eig_vals, eig_vecs = eig(A, b=B)

    return eig_vals, np.hsplit(eig_vecs, 2)[0]
