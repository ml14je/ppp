#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Fri Aug 28 15:14:23 2020
"""

import numpy as np

class alu(object):
    def __init__(self, M):
        self.M = np.copy(M).astype(float)
        assert len(self.M.shape) == 2
        assert self.M.shape[0] == self.M.shape[1]
        self.n = self.M.shape[0]
        self.lu_factor()

    def lu_factor(self):
        """
            LU factorization with partial pivorting
    
            Overwrite A with: 
                U (upper triangular) and (unit Lower triangular) L 
            Return [LU,piv] 
                Where piv is 1d numpy array with row swap indices 
        """
        self.lu = np.copy(self.M)
        self.indx = np.arange(0, self.n)
    
        for k in range(self.n-1):
            # pivot permutations
            max_row_index = np.argmax(abs(self.lu[k:self.n, k])) + k
            self.indx[[k, max_row_index]] = self.indx[[max_row_index, k]]
            self.lu[[k,max_row_index]] = self.lu[[max_row_index, k]]
    
            # Constructing LU matrix 
            for i in range(k+1, self.n):          
                self.lu[i, k] = self.lu[i, k]/self.lu[k, k]      
                for j in range(k+1, self.n):      
                    self.lu[i, j] -= self.lu[i, k]*self.lu[k, j] 
    
    def ufsub(self, b):
        """ Unit row oriented forward substitution """
        for i in range(self.n): 
            for j in range(i):
                b[i] -= self.lu[i, j]*b[j]
    
        return b

    def bsub(self, y):
        """ Row oriented backward substitution """
        for i in range(self.n-1,-1,-1): 
            for j in range(i+1, self.n):
                y[i] -= self.lu[i,j]*y[j]
            y[i] = y[i]/self.lu[i,i]
        return y
    
    def solve(self, b):
        b = b[self.indx].astype(float)

        return self.bsub(self.ufsub(b))
 
    
def test():
    from scipy.linalg import inv
    from time import perf_counter
    from ppp.Plots import plot_setup
    import matplotlib.pyplot as pt
    
    M = 100
    times = np.empty((2, M))
    for i in range(M):
        A = np.random.randint(10, size=(i+2, i+2))
        b = np.random.randint(10, size=(i+2, 1))
    
        start = perf_counter()
        inv(A) @ b
        times[0, i] = perf_counter() - start
        
        start = perf_counter()
        LU = alu(A)
        LU.solve(b)
        times[1, i] = perf_counter() - start
    
    fig, ax = plot_setup('Matrix Size', 'Time [s]')
    
    lineObjects = ax.plot(np.arange(M)+2, times.T)
    ax.legend(iter(lineObjects), ['LARPACK', 'LU Decomposition'],
              fontsize=16)

    pt.show()
    
if __name__ == '__main__':
    test()
    