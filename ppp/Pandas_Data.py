#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes : NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7 : Tue Nov  5 15:04:33 2019
"""
import os

def save_data(df, file_name, wd=None, folder_name=None):

    if not wd: wd = os.getcwd()
    if not folder_name: folder_name = ''

    df.to_csv(os.path.join(
        wd, os.path.join(
            folder_name, file_name+'.csv.gz'
            )
        ),
              index=False,
              compression='gzip')

def load_data(file_name, wd=None, folder_name=None):
    import pandas as pd

    if not wd: wd = os.getcwd()
    if not folder_name: folder_name = ''

    return pd.read_csv(os.path.join(
        wd, os.path.join(
            folder_name, file_name+'.csv.gz'
            )
        ),
                       compression='gzip')
