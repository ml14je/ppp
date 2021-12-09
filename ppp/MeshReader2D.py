#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Mon Jun 21 18:45:23 2021
"""
import numpy as np

def MeshReaderGambit(FileName):
    """
    function [Nv, VX, VY, K, EToV] = MeshReaderGambit2D(FileName)
    Purpose  : Read in basic grid information to build grid
    NOTE     : gambit(Fluent, Inc) *.neu format is assumed
    """

    with open(FileName, 'rt') as fid:
        # read intro
        lines = fid.readlines()
        for i in range(7):
            line = lines.pop(0)

        # Find number of nodes and number of elements
        dims = np.array(line.split(), int)
        Nv, K = dims[0], dims[1]

        for i in range(2):
            line = lines.pop(0)

        # read node coordinates
        VX, VY = np.zeros(Nv), np.zeros(Nv)
        for i in range(Nv):
            line = lines.pop(0)
            tmpx = np.array(line.split(), float)
            VX[i], VY[i] = tmpx[1], tmpx[2]


        for i in range(2):
          line = lines.pop(0)

        # read element to node connectivity
        EToV = np.zeros((K, 3), dtype=int)
        for k in range(K):
            line = lines.pop(0)
            tmpcon = np.array(line.split(), float)
            EToV[k, :3] = tmpcon[3], tmpcon[4], tmpcon[5]

    return Nv, VX, VY, K, EToV-1

def MeshReaderGambitBC(FileName):
    """
    function [Nv, VX, VY, K, EToV] = MeshReaderGambit2D(FileName)
    Purpose  : Read in basic grid information to build grid
    NOTE     : gambit(Fluent, Inc) *.neu format is assumed
    """

    with open(FileName, 'rt') as fid:
        # read intro
        lines = fid.readlines()
        for i in range(7):
            line = lines.pop(0)

        # Find number of nodes and number of elements
        dims = np.array(line.split(), int)
        Nv, K = dims[0], dims[1]

        for i in range(2):
            line = lines.pop(0)

        # read node coordinates
        VX, VY = np.zeros(Nv), np.zeros(Nv)
        for i in range(Nv):
            line = lines.pop(0)
            tmpx = np.array(line.split(), float)
            VX[i], VY[i] = tmpx[1], tmpx[2]


        for i in range(2):
          line = lines.pop(0)

        # read element to node connectivity
        EToV = np.zeros((K, 3), int)
        for k in range(K):
            line = lines.pop(0)
            tmpcon = np.array(line.split(), float)
            EToV[k, :3] = tmpcon[3]-1, tmpcon[4]-1, tmpcon[5]-1

        # skip through material property section
        for i in range(4):
            line = lines.pop(0)

        while True:
            line = lines.pop(0)
            if line.find('ENDOFSECTION') + 1:
                break


        try:
            for i in range(2):
                line = lines.pop(0)
        except IndexError:
            pass


        # Bundary codes (defined in Globals2D)
        BCType = np.zeros((K, 3), int)
        # Read all the boundary conditions at the nodes

        BCs = {
            'In': 1,
            'Out': 2,
            'Wall': 3,
            'Far': 4,
            'Cyl': 5,
            'Dirichlet': 6,
            'Neumann': 7,
            'Slip': 8
            }

        while len(lines) != 0:
            for key in BCs.keys():
                if line.find(key) != -1:
                    bcflag = BCs[key]

            line = lines.pop(0)
            while True:
                if not line.find('ENDOFSECTION'):
                    break
                tmpid = np.array(line.split(), int)
                BCType[tmpid[0]-1, tmpid[2]-1] = bcflag
                line = lines.pop(0)
            if len(lines) == 0:
                break
            line = lines.pop(0)
            line = lines.pop(0)

    return Nv, VX, VY, K, EToV, BCType

if __name__=='__main__':
    Nv, VX, VY, K, EToV, BCType = MeshReaderGambitBC('Test_Data/kovA02.neu')