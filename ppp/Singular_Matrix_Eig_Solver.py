#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes : NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7 : Wed Nov 20 11:40:57 2019
"""
import numpy as np

def gupti(A, B, τ=1e-2, δ1, δ2, )

if __name__ == '__main__':
    A = np.array(
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [1, 0, 0, 0, 0, 0, 0],
                [1, 2, 1, 1, 1, 1, 1],
                [1, 2, 3, 3, 3, 3, 3],
                [1, 2, 3, 2, 2, 2, 2],
                [1, 2, 3, 4, 3, 3, 3],
                [1, 2, 3, 4, 5, 5, 4]
            ])
    
    B = np.array(
            [
                [-2, -2, -2, -2, -2, -2, -2],
                [2, -1, -1, -1, -1, -1, -1],
                [2, 5, 5, 5, 5, 5, 5],
                [2, 5, 5, 4, 4, 4, 4],
                [2, 5, 5, 6, 5, 5, 5],
                [2, 5, 5, 6, 7, 7, 7],
                [2, 5, 5, 6, 7, 6, 6]
            ])
    
    print(np.linalg.det(B))
