#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk
Python 3.7: Wed Jul  1 11:43:21 2020
"""

from setuptools import setup

setup(name='Python Personal Package',
      version= '0.1',
      description= 'Personal Python package for the use throughout the NERC-funded PhD in Applied Mathematics at the University of Leeds.',
      url= 'http://github.com/ml14je/ppp',
      author= 'Joseph Elmes',
      author_email= 'ml14je@leeds.ac.uk',
      license='None',
      install_requires=[
          'wheel', 'numpy', 'scipy', 'pandas', 'matplotlib', 'opencv-python', 'pillow', 'pytube', 'sympy', 'netCDF4', 'plotly'
      ],
      zip_safe=False)
