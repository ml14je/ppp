#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Tue Jul 20 22:29:10 2021
"""
import numpy as np
from scipy.sparse import csr_matrix as sp
import os

def save_arrays(file_name, arrays, wd=None,
                folder_name=None):
    from scipy.sparse import save_npz

    for array in arrays:
        assert array.dtype ==  'float64' or 'complex128'

    if not wd: wd = os.getcwd()
    if not folder_name: folder_name = ''

    from ppp.File_Management import dir_assurer
    folder_name = os.path.join(
        wd, folder_name
        )

    dir_assurer(folder_name)

    save_npz(os.path.join(
            folder_name, file_name+'.npz'
            ),
                        *arrays
                        )

def load_arrays(file_name, wd=None, folder_name=None):
    from scipy.sparse import load_npz
    if not wd: wd = os.getcwd()
    if not folder_name: folder_name = ''

    loaded = load_npz(
        os.path.join(
            wd, os.path.join(
                folder_name, file_name+'.npz'
                )
            )
        )

    return loaded

if __name__ == '__main__':
    test_array = sp(np.random.rand(5, 2**10))
    test_vector = sp(np.random.rand(2**10))
    arrays = test_array, test_vector
    save_arrays('arrays', arrays)
    a, b = load_arrays('arrays')
    print(a)
    print()
    print(b)
