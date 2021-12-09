#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Thu Feb  4 14:18:01 2021
"""
import numpy as np

def BFGS(p, func, gtol=1e-8, it_max=200, STPMX = .5, TOLX=1e-8, eps=1e-16):
    func = func
    gtol = gtol
    it_max = it_max

    def dfunc(p0, eps):
        dvec = np.zeros(n)
        e = np.sqrt(eps)
        for i in range(n):
            p1, p2 = np.copy(p0), np.copy(p0)
            p1[i] += e
            p2[i] -= e
            f1, f2 = func(p1), func(p2)
            dvec[i] = (f1 - f2)/ (2 * e)

        return dvec

    n = len(p)
    fp, gp = func(p), dfunc(p, eps)
    hessin = np.eye(n)
    xi = -gp
    Sum = np.sum(p**2)
    stpmax = STPMX * max(np.sqrt(Sum), n)

    #Main loop
    from math import sqrt
    for its in range(it_max):
        fp, pnew, check = lnsrch(func, p, fp, gp, xi, stpmax)
        # The new function evaluation occurs in lnsrch; save the function value in fp for the
        # next line search. It is usually safe to ignore the value of check.
        xi = pnew - p #update line direction
        p = pnew

        test = 0
        for i in range(n):
            temp = abs(xi[i])/max(abs(p[i]), 1)
            if temp > test:
                test = temp

        if test < TOLX:
            return pnew

        dg = np.copy(gp) #save the old gradient
        gp = dfunc(p, eps)
        test = 0
        den = max(fp, 1)
        for i in range(n):
            temp = abs(gp[i]) * max(abs(p[i]), 1)/den
            if temp > test:
                test = temp

        if test < gtol:
            v = np.random.random(size=2)
            v = .5 * stpmax * np.sqrt(np.sum(v*v))
            pnew += v
            gp = dfunc(p, eps)
            continue

        dg = gp - dg #difference of gradients
        hdg = np.zeros(n)
        for j in range(n):
            hdg += hessin[:, j] * dg[j]

        fac = fae = sumdg = sumxi = 0 # Calculate dot products for the denominators.
        for i in range(n):
            fac += dg[i] * xi[i]
            fae += dg[i] * hdg[i]
            sumdg += dg[i]**2
            sumxi += xi[i]**2

        if (fac > sqrt(eps*sumdg*sumxi)): #Skip update if fac not sufficiently positive
            fac = 1.0/fac
            fad = 1.0/fae
            #The vector that makes BFGS different from DFP:
            dg = fac * xi - fad * hdg

            for i in range(n): # The BFGS updating formula:
                for j in range(n):
                    hessin[i][j] += fac * xi[i] * xi[j] - \
                        fad * hdg[i] * hdg[j] + fae * dg[i] * dg[j]

                    hessin[j][i] = hessin[i][j]

        # Now calculate the next direction to go,
        xi = np.zeros(n)
        for j in range(n):
            xi -= hessin[:, j] * gp[j]

    raise ValueError("too many iterations in dfpmin")


def lnsrch(func, xold, fold, g, p, stpmax):
    from math import sqrt
    f2, slope, S, alam2 = 0, 0, 0, 0
    # ALF ensures sufficient decrease in function value; TOLX is the convergence criterion on Δx.
    ALF, TOLX = 1e-4, 1e-16
    slope, S = 0, 0
    n = xold.shape[0]
    check = False

    for i in range(n):
        S += p[i] * p[i]
    S = sqrt(S)

    if S > stpmax:
        for i in range(n):
            p[i] *= stpmax/S

    for i in range(n):
        slope += g[i] * p[i]

    if slope >= 0:
        print('possible error')
        # raise ValueError("Roundoff problem in lnsrch.")

    test = 0
    for i in range(n):
        temp = abs(p[i])/max(abs(xold[i]), 1)
        if temp > test: test = temp

    alamin = TOLX/test
    alam = 1
    i = 0
    while True:
        x = xold + alam * p
        f = func(x)
        if alam < alamin:
            x = np.copy(xold)
            check = True

            return f, x, check

        elif f <= fold + ALF*alam*slope:
            return f, x, check

        else:
            if alam == 1:
                tmplam = -slope/(2*(f-fold-slope))

            else: #Subsequent backtracks
                rhs1, rhs2 = f-fold-alam*slope, f2-fold-alam2*slope
                a = (rhs1/(alam*alam)-rhs2/(alam2*alam2))/(alam-alam2)
                b = (-alam2*rhs1/(alam*alam)+alam*rhs2/(alam2*alam2))/(alam-alam2)

                if a == 0:
                    tmplam = -slope/(2*b)

                else:
                    disc = b * b - 3.0 * a * slope
                    if (disc < 0):
                        tmplam = .5 * alam # λ <= .5 * λ

                    elif b <= 0:
                        tmplam=(-b+sqrt(disc))/(3.0 * a)

                    else:
                        tmplam=-slope/(b+sqrt(disc))

                if tmplam > 0.5 * alam:
                    tmplam = .5 * alam # λ >= .1 * λ_1

            i += 1
        alam2 = alam
        f2 = f
        alam = max(tmplam, .1 * alam)

if __name__ == '__main__':
    p = np.array([2, 3], dtype=float)
    func = lambda x: np.cos(x[0]) * np.sin(x[1])

    print(BFGS(p, func))
