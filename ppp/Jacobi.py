#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Thu Feb 25 23:27:49 2021
"""
import numpy as np

def JacobiGL(alpha, beta, N):
    """
    Compute the N’th order Gauss Lobatto quadrature
    points, x, associated with the Jacobi polynomial,
    of type (alpha,beta) > -1 ( <> -0.5).
    """
    x = np.zeros((N+1, 1))
    x[0], x[-1] = -1, 1

    if N != 1:
        xint, w = JacobiGQ(alpha+1, beta+1, N-2)
        x[1:-1] = xint

    return x

def JacobiGQ(alpha, beta, N):
    """
    Compute the N’th order Gauss quadrature points, x,
    and weights, w, associated with the Jacobi
    polynomial, of type (alpha,beta) > -1 ( <> -0.5).
    """
    if N == 0:
        return (alpha-beta)/(alpha+beta+2), 2
    x, w = np.zeros((N-1, 1)), np.zeros((N-1, 1))
    J = np.zeros(N+1)
    h1 = 2*np.linspace(0, N, N+1) + alpha + beta
    v1 = np.linspace(1, N, N)

    with np.errstate(divide='ignore', invalid='ignore'):
        J = np.diag(-1/2*(alpha**2-beta**2)/(h1+2)/h1) + \
            np.diag(2/(h1[:-1]+2)*np.sqrt(v1*(v1+alpha+beta) * \
                  (v1+alpha)*(v1+beta)/(h1[:-1]+1)/(h1[:-1]+3)), k=1)

    if alpha + beta < 1e-10:
        J[0, 0] = 0

    J += J.T
    from scipy.linalg import eig
    from scipy.special import gamma
    x, V = eig(J)
    inds = np.argsort(x)
    x, V = x[inds], V[:, inds]
    w = (V[0,:].T**2) * 2**(alpha+beta+1)/(alpha+beta+1) * gamma(alpha+1) * \
    gamma(beta+1)/gamma(alpha+beta+1)

    return x.real[:, None], w[:, None]

def JacobiP(x, alpha, beta, N):
    """
    Purpose: Evaluate Jacobi Polynomial of type (alpha,beta) > -1
    (alpha+beta <> -1) at points x for order N and returns
    P[1:length(xp))]
    Note : They are normalized to be orthonormal.
    Turn points into row if needed.
    """
    xp = x
    dims = xp.shape
    #Turn points into row if needed.
    try:
        if dims[1]==1:
            xp = xp.T
    except IndexError:
        xp = xp[None,:]
        assert xp.shape[0] == 1

    from scipy.special import gamma
    PL = np.zeros((N+1, xp.shape[1]))
    #Initial values P_0(x) and P_1(x)
    gamma0 = 2**(alpha+beta+1)/(alpha+beta+1)*gamma(alpha+1)* \
        gamma(beta+1)/gamma(alpha+beta+1)
    PL[0] = 1/np.sqrt(gamma0)

    if N == 0:
        return PL[N].T

    else:
        gamma1 = (alpha+1)*(beta+1)/(alpha+beta+3)*gamma0
        PL[1] = ((alpha+beta+2)*xp/2 + (alpha-beta)/2)/np.sqrt(gamma1)
        if N != 1:
            #Repeat value in recurrence.
            aold = 2/(2+alpha+beta)*np.sqrt((alpha+1)*(beta+1)/(alpha+beta+3))
            #Forward recurrence using the symmetry of the recurrence.
            for i in range(1, N):
                h1 = 2*i+alpha+beta
                anew = 2/(h1+2)*np.sqrt((i+1)*(i+1+alpha+beta)*(i+1+alpha)*\
                                        (i+1+beta)/(h1+1)/(h1+3))
                bnew = - (alpha**2-beta**2)/h1/(h1+2)
                PL[i+1] = 1/anew*( -aold*PL[i-1] + (xp-bnew)*PL[i])
                aold =anew

        return PL[N].T

def GradJacobiP(r, alpha, beta, N):
    """
    Evaluate the derivative of the Jacobi polynomial of type
    (alpha,beta)>-1, at points r for order N and returns
    dP[1:len(r))]
    """
    dP = np.zeros(len(r))

    if N != 0:
        dP = np.sqrt(N*(N+alpha+beta+1)) * JacobiP(r, alpha+1, beta+1, N-1)

    return dP

def Simplex2DP(a, b, i, j):
    """
    Purpose : Evaluate 2D orthonormal polynomial
    on simplex at (a,b) of order (i,j).
    """
    h1 = JacobiP(a[:, None], 0, 0, i)
    h2 = JacobiP(b[:, None], 2*i+1, 0, j)

    return np.sqrt(2) * h1 * h2 * (1-b)**i

def Simplex3DP(a, b, c, i, j, k):
    """
    Purpose : Evaluate 3D orthonormal polynomial
    on simplex at (a,b,c) of order (i,j,k).
    """
    h1 = JacobiP(a, 0, 0, i)
    h2 = JacobiP(b, 2*i+1, 0, j)
    h3 = JacobiP(c, 2*(1+j+1), 0, k)

    return 2 * np.sqrt(2) * h1 * h2 * ((1-b)**i) * h3 * ((1-c)**(i+j))

def GradSimplex2DP(a, b, idd, jdd):
    """
    Return the derivatives of the modal basis (id,jd)
    on the 2D simplex at (a,b).
    """
    fa, dfa= JacobiP(a, 0, 0, idd),  GradJacobiP(a, 0, 0, idd)
    gb, dgb = JacobiP(b, 2*idd+1, 0, jdd), GradJacobiP(b, 2*idd+1, 0, jdd)
    #r-derivative
    #d/dr = da/dr d/da + db/dr d/db = (2/(1-s)) d/da = (2/(1-b)) d/da
    dmodedr = dfa * gb
    if idd > 0 :
        dmodedr = dmodedr * ((.5 * (1-b))**(idd-1))

    #s-derivative
    #d/ds = ((1+a)/2)/((1-b)/2) d/da + d/db
    dmodeds = dfa * (gb * (.5 * (1+a)))
    if idd > 0:
        dmodeds = dmodeds * ((.5*(1-b))**(idd-1))
    tmp = dgb * ((.5 * (1-b))**idd);
    if idd > 0:
        tmp -= .5 * idd * gb * ((.5 * (1-b))**(idd-1))
    dmodeds += fa * tmp
    #Normalize
    dmodedr *= 2**(idd + .5)
    dmodeds *= 2**(idd + .5)
    # print('dmodeds', dmodeds)

    return dmodedr, dmodeds

if __name__ == '__main__':
    x = np.linspace(0, 1, 11)[:, None]
    print(JacobiP(x, 0, 0, 3))
