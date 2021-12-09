#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes : NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7 : Tue Nov  5 14:20:41 2019
"""

import numpy as np
import os

def save_arrays(file_name, arrays, wd=None,
                folder_name=None):

    for array in arrays:
        assert array.dtype ==  'float64' or 'complex128'

    if not wd: wd = os.getcwd()
    if not folder_name: folder_name = ''

    from ppp.File_Management import dir_assurer
    folder_name = os.path.join(
        wd, folder_name
        )
    dir_assurer(folder_name)

    np.savez_compressed(os.path.join(
            folder_name, file_name+'.npz'
            ),
                        *arrays
                        )

def load_arrays(file_name, wd=None, folder_name=None):

    if not wd: wd = os.getcwd()
    if not folder_name: folder_name = ''

    loaded = np.load(
        os.path.join(
            wd, os.path.join(
                folder_name, file_name+'.npz'
                )
            ), allow_pickle=True
        )

    return tuple([loaded[key] for key in loaded])

if __name__ == '__main__':
    test_array = np.random.rand(5, 2**10)
    test_vector = np.random.rand(2**10)
    arrays = test_array, test_vector
    save_arrays('arrays', arrays)
    a, b = load_arrays('arrays')
    print(a)
    print()
    print(b)
