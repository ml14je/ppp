#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Wed Feb 24 23:30:59 2021
"""

def crossings_nonzero_pos2neg(data):
        pos = data > 0
        return (pos[:-1] & ~pos[1:]).nonzero()[0]

