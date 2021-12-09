#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes : NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7 : Tue Nov  5 10:37:45 2019
"""

def dir_assurer(folder_name, wd=''):
    import os

    if not os.path.exists(
        os.path.join(
            wd, folder_name
            )
        ):
        os.makedirs(
            os.path.join(
                wd, folder_name
                )
            )
            
def file_exist(file_name, folder_name='', wd=''):
    import os

    return os.path.isfile(
        os.path.join(
            wd, os.path.join(
                folder_name, file_name
                )
            )
        )
            
def obtain_files(wd, folder_name):
    import os

    return os.listdir(
        os.path.join(
            wd, folder_name
            )
        )

if __name__ == '__main__':
    dir_assurer('a/bb/c/d/e')
