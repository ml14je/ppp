#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes : NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7 : Wed Nov  6 09:51:22 2019
"""
import sys
mod_path = "C:\\Users\\home\\OneDrive - University of Leeds\\Documents\\PhD"
if mod_path not in sys.path:
    sys.path.insert(
        0, mod_path
        )

import numpy as np

def A_matrix(I, A0, A1, A2):
    O = 0*I
    return np.block([
        [O, I, O],
        [O, O, I],
        [A0, A1, A2]
        ])  

def B_matrix(I, A3):
    O = 0*I
    return np.block([
        [I, O, O],
        [O, I, O],
        [O, O, -A3]
        ])

def cubic_evp(I, A0, A1, A2, A3):
    from scipy.linalg import eig
    import numpy as np

    A = A_matrix(I, A0, A1, A2) ; B = B_matrix(I, A3)
    eig_vals, eig_vecs = eig(A, b=B)

    return eig_vals, np.hsplit(eig_vecs, 3)[0]
