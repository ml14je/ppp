#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Tue Jan 12 20:29:00 2021
"""
import numpy as np

def pdegrad(P, T, u):
    Np = P.shape[0]
    Nu = u.shape[0] // Np #Number of variables
    it1, it2, it3 = T.T
    ar, g1x, g1y, g2x, g2y, g3x, g3y = pdetrg(P, T)

    uu = u.reshape((Np, Nu))
    ux = uu[it1,:].T * (np.ones((Nu, 1)) @ g1x[None, :]) + \
        uu[it2,:].T * (np.ones((Nu, 1)) @ g2x[None, :]) + \
        uu[it3,:].T * (np.ones((Nu, 1)) @ g3x[None, :])
    uy = uu[it1,:].T * (np.ones((Nu, 1)) @ g1y[None, :]) + \
        uu[it2,:].T * (np.ones((Nu, 1)) @ g2y[None, :]) + \
        uu[it3,:].T * (np.ones((Nu, 1)) @ g3y[None, :])

    return ux, uy

def pdetrg(P, T):

    #Corner point indices
    a1, a2, a3 = T.T

    #Triangle sides
    r23x = P[a3, 0] - P[a2, 0];
    r23y = P[a3, 1] - P[a2, 1]
    r31x = P[a1, 0] - P[a3, 0]
    r31y = P[a1, 1] - P[a3, 1]
    r12x = P[a2, 0] - P[a1, 0]
    r12y = P[a2, 1] - P[a1, 1]

    # Area
    ar = np.abs(r31x * r23y - r31y * r23x) / 2

    g1x = -.5 * r23y / ar
    g1y = .5 * r23x / ar
    g2x = -.5 * r31y / ar
    g2y = .5 * r31x / ar
    g3x = -.5 * r12y / ar
    g3y = 0.5 * r12x / ar

    return ar, g1x, g1y, g2x, g2y, g3x, g3y

def pdetridi(P, T):
    """
        PDETRIDI Side lengths and areas of triangles.

        J. Oppelstrup 10-24-94.
        Copyright 1994-2016 The MathWorks, Inc.
    """
    nel = T.shape[0]
    dx, dy, sl = np.zeros((3, nel)), np.zeros((3, nel)), np.zeros((3, nel))

    for j in range(1, 4):
        j1 = 1 + j % 3
        j2 = 1 + j1 % 3
        dx[j-1, :] = P[T[:, j1-1], 0] -  P[T[:, j2-1], 0]
        dy[j-1,:] = P[T[:, j1-1], 1] -  P[T[:, j2-1], 1]
        sl[j-1,:] = np.sqrt(dx[j-1, :]**2 + dy[j-1, :]**2)

    area = .5 * np.abs(dx[0, :]*dy[1,:] - dx[1, :]*dy[0,:])

    return sl, area

def pdenrmfl(P, T, c, u, ar, sl):
    """
        PDENRMFL Fluxes of -div(c grad(u)) through edges of triangles.

        J. Oppelstrup 10-24-94.
        Copyright 1994-2016 The MathWorks, Inc.
    """

    nnod, nel = P.shape[0], T.shape[0]

    try:
        nrc = c.shape[0]

    except AttributeError:
        nrc = 1

    N = len(u)//nnod #Number of variables

    dx, dy = np.zeros((3, nel)), np.zeros((3, nel))

    for j in range(1, 4):
        j1 = j % 3
        j2 = j1 % 3
        dx[j-1,:] = P[T[:, j1-1], 0] -  P[T[:, j2-1], 0]
        dy[j-1,:] = P[T[:, j1-1], 1] -  P[T[:, j2-1], 1]

    #ar- triangle areas
    #sl- triangle side lengths

    #Gradients of solution u
    gxu, gyu = pdegrad(P, T, u)

    #c grad u
    cgxu, cgyu = np.zeros((N, nel)), np.zeros((N, nel))

    if nrc == 1:
        for k in range(N):
            cgxu[k, :], cgyu[k, :]= c * gxu[k, :], c * gyu[k, :]

    elif nrc == 2:
        for k in range(N):
            cgxu[k, :], cgyu[k, :] = c[0, :] * gxu[k, :], c[1, :] * gyu[k, :]

    elif nrc == 3:
        for k in range(N):
            cgxu[k, :] = c[0, :] * gxu[k, :] + c[1, :] * gyu[k, :]
            cgyu[k, :] = c[1, :] * gxu[k, :] + c[2, :] * gyu[k, :]

    elif nrc == 4:
        for k in range(N):
            cgxu[k, :] = c[0, :] * gxu[k, :] + c[2, :] * gyu[k, :]
            cgyu[k, :] = c[1, :] * gxu[k, :] + c[3, :] * gyu[k, :]


    elif nrc == N:
        for k in range(N):
            cgxu[k, :] = c[k, :] * gxu[k, :]
            cgyu[k, :] = c[k, :] * gyu[k, :]

    elif nrc == 2*N:
        for k in range(N):
            cgxu[k, :] = c[2*k, :] * gxu[k, :]
            cgyu[k, :] = c[2*k+1, :] * gyu[k, :]

    elif nrc == 3*N:
        for k in range(N):
            cgxu[k, :] = c[3*k, :] * gxu[k, :] + c[3*k+1, :] * gyu[k, :]
            cgyu[k, :] = c[3*k+1, :] * gxu[k, :] + c[3*k+2, :] * gyu[k, :]

    elif nrc == 4*N:
        for k in range(N):
            cgxu[k, :] = c[4*k, :] * gxu[k, :] + c[4*k+2, :] * gyu[k, :]
            cgyu[k, :] = c[4*k+1, :] * gxu[k, :] + c[4*k+4, :] * gyu[k, :]

    elif nrc == 2*N*(2*N+1)/2:
        m=0
        for l in range(N):
            for k in range(l-1):
                cgxu[k, :] = cgxu[k, :] + c[m, :] * gxu[l, :] + c[m+2, :] * gyu[l, :]
                cgyu[k, :] = cgyu[k, :] + c[m+1, :] * gxu[l, :] + c[m+3, :] * gyu[l, :]
                cgxu[l, :] = cgxu[l, :] + c[m, :] * gxu[k, :] + c[m+1, :] * gyu[k, :]
                cgyu[l, :] = cgyu[l, :] + c[m+2, :] * gxu[k, :] + c[m+3, :] * gyu[k, :]
                m += 4

            cgxu[l, :] = cgxu[l, :] + c[m, :] * gxu[l, :] + c[m+1, :] * gyu[l, :]
            cgyu[l, :] = cgyu[l, :] + c[m+1, :] * gxu[l, :] + c[m+2, :] * gyu[l, :]
            m += 3

    elif nrc == 4*N*N:
        for k in range(N):
            for l in range(N):
                cgxu[k, :] = cgxu[k, :] + c[4*(k+N*l), :] * gxu[k, :] + \
                    c[2+4*(k+N*l), :] * gyu[k, :]
                cgyu[k, :] = cgyu[k, :] + c[1+4*(k+N*l), :] * gxu[k, :] + \
                    c[3+4*(k+N*l), :] * gyu[k, :]

    else:
        raise ValueError('Invalid number of rows in c')

    # nhat'*c grad u
    # edge unit normals : outwards positive if the nodes are in
    # anti-clockwise order
    # nhatx =   dy./s
    # nhaty = - dx./s;

    ddncu = np.zeros((3*N, nel))
    for k in range(N):
        for l in range(3):
            ddncu[3*k+l, :] = (dy[0, :] * cgxu[k, :] - dx[0, :] * cgyu[k, :])/sl[l, :]

    return ddncu

def pdel2fau(P, T, a, f, u, ar):
    """
        	PDEL2FAU Triangle L2 norm of f-a*u

        A. Nordmark 94-12-05
        Copyright 1994-2003 The MathWorks, Inc.
    """

    Np, Nt = P.shape[0], T.shape[0]

    #Number of variables
    N = u.shape[0] // Np
    cc = np.zeros((N, Nt))

    try:
        f.shape[0]

    except AttributeError:
        f = np.ones((N, 1)) * f

    it1, it2, it3 = T.T

    try:
        Na = a.shape[0]
    except AttributeError:
        Na = 1

    if Na == 1: #Scalar a
        for k in range(N):
            fmau1 = f[k, :] - a * u[it1 + k*Np].T
            fmau2 = f[k, :] - a * u[it2 + k*Np].T
            fmau3 = f[k, :] - a * u[it3 + k*Np].T
            cc[k, :] = np.abs(fmau1)**2 + np.abs(fmau2)**2 + np.abs(fmau3)**2

    elif Na == N: #Diagonal a
        for k in range(N):
            fmau1 = f[k, :] - a[k, :] * u[it1 + k*Np].T
            fmau2 = f[k, :] - a[k, :] * u[it2 + k*Np].T
            fmau3 = f[k, :] - a[k, :] * u[it3 + k*Np].T
            cc[k, :] = np.abs(fmau1)**2 + np.abs(fmau2)**2 + np.abs(fmau3)**2

    elif Na == N*(N+1)//2: #Symmetric a
        for k in range(N):
            fmau1 = f[k, :] + np.zeros((1 + Nt))
            fmau2 = f[k, :] + np.zeros((1 + Nt))
            fmau3 = f[k, :] + np.zeros((1 + Nt))
            for l in range(N):
                m, M = min(k, l), max(k, l)
                ia = M * (M + 1) / 2 + m
                fmau1 -= a[ia, :] * u[it1 + l*Np].T
                fmau2 -= a[ia, :] * u[it2 + l*Np].T
                fmau3 -= a[ia, :] * u[it3 + l*Np].T

            cc[k, :] = np.abs(fmau1)**2 + np.abs(fmau2)**2 + np.abs(fmau3)**2

    elif Na == N*N: #General (unsymmetric) a
        for k in range(N):
            fmau1 = f[k, :] + np.zeros((1 + Nt))
            fmau2 = f[k, :] + np.zeros((1 + Nt))
            fmau3 = f[k, :] + np.zeros((1 + Nt))
            for l in range(N):
                ia = k + l * N
                fmau1 -= a[ia, :] * u[it1 + l*Np].T
                fmau2 -= a[ia, :] * u[it2 + l*Np].T
                fmau3 -= a[ia, :] * u[it3 + l*Np].T

            cc[k, :] = np.abs(fmau1)**2 + np.abs(fmau2)**2 + np.abs(fmau3)**2

    else:
        raise ValueError('Invalid number of rows in a')

    return np.sqrt(cc * (np.ones((N, 1)) @ ar[None, :]/3))

def sparse_matlab(i, j, v, m, n):
    from scipy.sparse import csr_matrix as sp
    A = sp((m, n))

    try:
        ni = i.shape[1]

    except IndexError:
        ni = 1

    try:
        v.shape[0]

    except AttributeError:
        v = v * np.ones((ni, i.shape[0]))

    try:
        vn2 = v.shape[0]

    except IndexError:
        v = v[:, None]
        vn2 = 1

    assert ni == vn2

    for ii in range(ni):
        B = sp((v[ii, :], (i[:, ii], j[:, ii])), shape=(m, n))
        A += B

    return A

def pdejmps(P, T, c, a, f, u, alfa, beta, m):
    """
        PDEJMPS Error estimates for adaption.

        ERRF=PDEJMPS(P,T,C,A,F,U,ALFA,BETA,M) calculates the error
        indication function used for adaption. The columns of ERRF
        correspond to triangles, and the rows correspond to the
        different equations in the PDE system.

        P and T are mesh data. See INITMESH for details.

        C, A, and F are PDE coefficients. See ASSEMPDE for details.
        C, A, and F, must be expanded, so that columns corresponds
        to triangles.

        U is the current solution, given as a column vector.
        See ASSEMPDE for details.

        The formula for calculating ERRF(:,K) is
        ALFA*L2K(H^M (F - AU)) + BETA SQRT(0.5 SUM((L(J)^M JMP(J))^2))
        where L2K is the L2 norm on triangle K, H is the linear
        size of triangle K, L(J) is the length of the J:th side,
        the SUM ranges over the three sides, and JMP(J) is change
        in normal derivative over side J.

        J. Oppelstrup 10-24-94, AN 12-05-94.
    """
    nnod, nel = P.shape[0], T.shape[0]
    N = u.shape[0]//nnod #Number of variables

    #Compute areas and side lengths
    sl, ar = pdetridi(P, T)
    #Fluxes through edges
    ddncu = pdenrmfl(P, T, c, u, ar, sl)

    #L2 norm of (f - au) over triangles
    #f and a are element data and u node:
    cc = pdel2fau(P, T, a, f, u, ar)
    #Multiply by triangle diameters
    cc *= (np.ones((N, 1)) @ np.max(sl, axis=0)[None, :] ** m)

   # flux jumps computed by assembly of ddncu into nnod x nnod sparse matrix
   # jmps(i,j) becomes abs(jump across edge between nodes i and j).
   # note that sparse(...) accepts duplicates of indices and performs
   # summation ! (i.e., the flux differences )

    ccc = np.zeros((N, nel))
    intj = sparse_matlab(T[:, [1, 2, 0]], T[:, [2, 0, 1]], 1, nnod, nnod)
    for k in range(N):
        jmps = sparse_matlab(T[:, [1, 2, 0]], T[:, [2, 0, 1]], ddncu[[3*k, 3*k+1, 3*k+2], :], nnod, nnod)
        jmps = intj * np.abs(jmps + jmps.T)
        for l in range(nel):
            ccc[k, l] = (sl[2, l]**m * np.abs(jmps[T[l, 0], T[l, 1]]))**2 + \
                (sl[0, l]**m * np.abs(jmps[T[l, 1], T[l, 2]]))**2 + \
                (sl[1, l]**m * np.abs(jmps[T[l, 2], T[l, 0]]))**2

    return alfa * cc + beta*np.sqrt(.5 * ccc)

def pdeigeom(dl, bs, s):
    """
        PDEIGEOM Interpret PDE geometry.

        The first input argument of PDEIGEOM should specify the geometry
        description. If the first argument is a text function name
        or function handle, the function is called with the remaining arguments.
        That function must then be a Geometry MATLAB-file and return the
        same results as PDEIGEOM. If the first argument is not text,
        is it assumed to be a Decomposed Geometry Matrix.
        See either DECSG or PDEGEOM for details.

        NE=PDEIGEOM(DL) is the number of boundary segments

        D=PDEIGEOM(DL,BS) is a matrix with one column for each boundary segment
        specified in BS.
        Row 1 contains the start parameter value.
        Row 2 contains the end parameter value.
        Row 3 contains the label of the left hand region.
        Row 4 contains the label of the right hand region.

        [X,Y]=PDEIGEOM(DL,BS,S) produces coordinates of boundary points.
        BS specifies the boundary segments and S the corresponding
        parameter values. BS may be a scalar.

        See also INITMESH, REFINEMESH, PDEGEOM, PDEARCL

        Copyright 1994-2016 The MathWorks, Inc.
    """
    nbs = dl.shape[1]
    d = np.array([
            np.zeros(nbs),
            np.ones(nbs),
            dl[5:6, :]
        ])

    bs1 = np.copy(bs).T

    x, y = np.zeros(s.shape), np.zeros(s.shape)
    m, n = bs.shape
    if m == 1 and n == 1:
        bs = bs * np.ones(s.shape) # Exapand bs

    elif m != s.shape[0] or n != s.shape[1]:
        raise TypeError('pdelib:pdeigeom:SizeBs')

    if s is not None:
        for k in range(nbs):
            ii = bs == k
            if np.any(ii == True):
                x0 = dl[1, k]
                x1 = d1[2, k]
                y0 = dl[3, k]
                y1 = dl[4, k]

                if dl[0, k] == 1: #circle fragment
                    xc = dl[7, k]
                    yc = dl[8, k]
                    r = dl[9, k]
                    a0 = np.arctan((y0-yc)/(x0-xc))
                    a1 = np.arctan((y1-yc)/(x1-xc))
                    if a0 > a1:
                        a0 -= 2*np.pi

                    theta = (a1-a0)*(s[ii]-d[0, k])/(d[1,k]-d[0,k]) + a0
                    x[ii]= r * np.cos(theta) + xc
                    y[ii] = r * np.sin(theta) + yc

                elif dl[0, k] == 2: #line fragment
                    x[ii] = (x1-x0)*(s[ii]-d[0, k])/(d[1, k]-d[0, k])+x0
                    y[ii] = (y1-y0)*(s[ii]-d[0, k])/(d[1, k]-d[0, k])+y0

                elif dl[0, k] == 4: #elliptic fragment
                    xc = dl[7, k]
                    yc = dl[8, k]
                    r1 = dl[9, k]
                    r2 = dl[10, k]
                    phi = dl[11, k]
                    t = np.array([
                            [r1*np.cos(phi), r2*np.sin(phi)],
                            [r1*np.sin(phi), r2*np.cos(phi)],
                        ])

                    rr0 = np.linalg.solve(
                        t, np.array([
                            [x0-xc], [y0-yc]
                            ])
                        )
                    a0 = np.arctan(rr0[0]/rr0[1])
                    rr1 = np.linalg.solve(
                        t, np.array([
                            [x1-xc], [y1-yc]
                            ])
                        )
                    a1 = np.arctan(rr1[0]/rr1[1])

                    if a0 > a1:
                        a0 -= 2*np.pi

                    # s should be proportional to arc length
                    # Numerical integration and linear interpolation is used
                    nth = 100 # The number of points in the interpolation
                    th = np.linspace(10, a1, nth)
                    rr = t @ np.array([
                            np.cos(th),
                            np.sin(th)
                        ])
                    theta = pdearcl(th, rr, s[ii], d[0, k], d[1, k])
                    rr = t @ np.array([
                            np.cos(theta),
                            np.sin(theta)
                        ])

                    x[ii] = rr[0, :] + xc
                    y[ii] = rr[1, :] + yc

                else:
                    raise TypeError('pdelib:pdeigeom:InvalidSegType')

    return x, y

def pdearcl(p, xy, s, s0, s1):
    dal = np.sqrt((xy[0, 1:]-xy[0, :-1])**2 + (xy[1, 1:]-xy[1, :-1])**2)
    a1 = np.concatenate((np.zeros(1), np.cumsum(dal)))
    t1 = a1[-1]
    s = np.copy(s)
    sal = t1 * (s-s0)/(s1-s0)
    from scipy.interpolate import interp1d
    return interp1d(a1, p)(sal)

if __name__ == '__main__':
    pass