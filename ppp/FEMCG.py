#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Sun Jan 10 00:42:47 2021
"""
import numpy as np
from scipy.sparse import csr_matrix as sp

class FEM_P1CG(object):
    def __init__(self, P, T, E, B, calc_area=True):#,
        self.P, self.T, self.E, self.B = P, T, E, B
        self.x, self.y = P.T
        self.xc = np.mean(self.x[:, None][T], axis=1)
        self.yc = np.mean(self.y[:, None][T], axis=1)
        self.Np, self.Nt, self.Ne = P.shape[0], T.shape[0], E.shape[0]
        self.Np0 = self.Np
        self.area()

    def area(self):
        self.K = np.zeros(self.Nt)
        for i in range(self.Nt):
            loc2glb = self.T[i]
            A = self.P[loc2glb]
            det = (A[1, 0] * A[2, 1] - A[2, 0] * A[1, 1]) - (A[0, 0] * A[2, 1] - A[2, 0] * A[0, 1]) + \
                        (A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1])
            self.K[i] = .5 * det

        self.A = np.sum(self.K)

    def IsoStiffnessAssembler(self, func):
        rspts, qwgts = GaussPoints(2) #Quadrature Rule
        S = sp((self.Np, self.Np)).tolil()
        for i in range(self.Nt):
            loc2glb = self.T[i] #node numbers
            x, y = self.P[loc2glb].T #Node x- and y-coordinates
            SK = sp((3, 3)) #elements stiffness
            for q in range(len(qwgts)): #quadrature loop
                r = rspts[q, 0] #quadrature r-coordinate
                s = rspts[q, 1] #quadrature s-coordinate
                Sv, dSdx, dSdy, detJ = Isopmap(x, y, r, s, self.shapes)
                xc, yc = Sv[:, 0].dot(x), Sv[:, 0].dot(y)
                wxarea = qwgts[q] * detJ/2 #weight times area
                SK +=  (dSdx @ dSdx.T + \
                        dSdy @ dSdy.T) * wxarea * func(xc, yc) #Element Stiffness

            S[loc2glb[:, None], loc2glb[None,:]] += SK

        return S.tocsr()

    def IsoMassAssembler(self):
        rspts, qwgts = GaussPoints(2) #Quadrature Rule
        M = sp((self.Np, self.Np)).tolil()
        for i in range(self.Nt):
            loc2glb = self.T[i] #node numbers
            x, y = self.P[loc2glb].T #Node x- and y-coordinates
            MK = sp((3, 3)) #elements stiffness
            for q in range(len(qwgts)): #quadrature loop
                r = rspts[q, 0] #quadrature r-coordinate
                s = rspts[q, 1] #quadrature s-coordinate
                S, dSdx, dSdy, detJ = Isopmap(x, y, r, s, self.shapes)
                wxarea = qwgts[q] * detJ/2 #weight times area
                MK +=  (S @ S.T) * wxarea#Element Stiffness
            M[loc2glb[:, None], loc2glb[None,:]] += MK

        return M.tocsr()

    def IsoStrainXAssembler(self, func):
        rspts, qwgts = GaussPoints(2) #Quadrature Rule
        S = sp((self.Np, self.Np)).tolil()
        for i in range(self.Nt):
            loc2glb = self.T[i] #node numbers
            x, y = self.P[loc2glb].T #Node x- and y-coordinates
            SK = sp((3, 3)) #elements stiffness
            for q in range(len(qwgts)): #quadrature loop
                r = rspts[q, 0] #quadrature r-coordinate
                s = rspts[q, 1] #quadrature s-coordinate
                Sv, dSdx, dSdy, detJ = Isopmap(x, y, r, s, self.shapes)
                xc, yc = Sv[:, 0].dot(x), Sv[:, 0].dot(y)
                wxarea = qwgts[q] * detJ/2 #weight times area
                SK +=  (Sv @ dSdx.T) * wxarea * func(xc, yc) #Element Stiffness

            S[loc2glb[:, None], loc2glb[None,:]] += SK

        return S.tocsr()

    def IsoStrainYAssembler(self, func):
        rspts, qwgts = GaussPoints(2) #Quadrature Rule
        S = sp((self.Np, self.Np)).tolil()
        for i in range(self.Nt):
            loc2glb = self.T[i] #node numbers
            x, y = self.P[loc2glb].T #Node x- and y-coordinates
            SK = sp((3, 3)) #elements stiffness
            for q in range(len(qwgts)): #quadrature loop
                r = rspts[q, 0] #quadrature r-coordinate
                s = rspts[q, 1] #quadrature s-coordinate
                Sv, dSdx, dSdy, detJ = Isopmap(x, y, r, s, self.shapes)
                xc, yc = Sv[:, 0].dot(x), Sv[:, 0].dot(y)
                wxarea = qwgts[q] * detJ/2 #weight times area
                SK +=  (Sv @ dSdy.T) * wxarea * func(xc, yc) #Element Stiffness

            S[loc2glb[:, None], loc2glb[None,:]] += SK

        return S.tocsr()

    def mass_matrix_assembly2D(self):
        M = sp((self.Np, self.Np)).tolil()

        for i in range(self.Nt):
            loc2glb = self.T[i]
            MK = np.array([
                [2, 1, 1],
                [1, 2, 1],
                [1, 1, 2]
                ]) * self.K[i]/12

            M[loc2glb[:, None], loc2glb[None,:]] += MK
        return M

    def load_matrix_assembly2D(self, func):
        b = np.zeros((self.Np, 1))

        for i in range(self.Nt):
            loc2glb = self.T[i]
            x, y = self.P[loc2glb].T

            bK = np.array([
                [func(x[0], y[0]), func(x[1], y[1]), func(x[2], y[2])]
                ]).T * self.K[i] / 3

            b[loc2glb] += bK

        return b

    def stiffness_matrix_assembly2D(self, func):
        S = sp((self.Np, self.Np)).tolil()

        for i in range(self.Nt):
            loc2glb = self.T[i]
            x, y = self.P[loc2glb].T
            b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])/(2*self.K[i])
            c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])/(2*self.K[i])
            xc, yc = np.mean(x), np.mean(y)
            abar = func(xc, yc)

            SK = abar * self.K[i] * (b[:, None]*b[None, :] + c[:, None]*c[None, :])
            S[loc2glb[:, None], loc2glb[None,:]] += SK

        return S.tocsr()

    def RobinMassMatrix2D(self, kappa):
        R = sp((self.Np, self.Np)).tolil()

        for i in range(self.Ne):
            loc2glb = self.E[i]
            x, y = self.P[loc2glb].T
            L = np.sqrt((x[0]-x[1])**2 + (y[0]-y[1])**2)
            xc, yc = np.mean(x), np.mean(y)
            k = kappa(xc, yc)
            RE = k * L * np.array([[2, 1], [1, 2]]) / 6
            R[loc2glb[:, None], loc2glb[None,:]] += RE

        return R.tocsr()

    def RobinLoadVector2D(self, kappa, gD, gN):
        r = np.zeros((self.Np, 1))
        for i in range(self.Ne):
            loc2glb = self.E[i]
            x, y = self.P[loc2glb].T
            L = np.sqrt((x[0]-x[1])**2 + (y[0]-y[1])**2)
            xc, yc = np.mean(x), np.mean(y)
            tmp = kappa(xc, yc) * gD(xc, yc) + gN(xc, yc)
            rE = tmp * np.ones((2, 1)) * L / 2
            r[loc2glb] += rE

        return r

    def Robin_assembly2D(self, kappa, gD, gN):
        R = self.RobinMassMatrix2D(kappa)
        r = self.RobinLoadVector2D(kappa, gD, gN)

        return R, r

    def show_mesh2D(self):
        import matplotlib.pyplot as pt
        print('hi there')
        from ppp.Plots import plot_setup
        fig, ax = plot_setup('$x$', '$y$')
        ax.triplot(self.P[:,0], self.P[:,1], self.T)
        ax.plot(self.P[:,0], self.P[:,1], 'o')
        ax.plot(self.B[:, 0], self.B[:, 1], 'bx')
        pt.show()

    def draw_boundary(self, ax):
        for e in self.E:
            ax.plot(self.x[e], self.y[e], 'k-')

    def shapes(self, r, s):
        S = np.array([
            [1-r-s], [r], [s]
            ])

        dSdr = np.array([
            [-1], [1], [0]
            ])

        dSds = np.array([
            [-1], [0], [1]
            ])

        return S, dSdr, dSds

class FEM_P2CG(object):
    def __init__(self, P, T, E, B):
        self.P, self.T, self.E, self.B = P, T, E, B
        self.Np, self.Nt, self.Ne = P.shape[0], T.shape[0], E.shape[0]
        self.changeP1toP2Mesh()
        self.x, self.y = self.P.T
        self.xc = np.mean(self.x[:, None][T], axis=1)
        self.yc = np.mean(self.y[:, None][T], axis=1)

    def shapes(self, r, s):
        S = np.array([
            [1-3*r-3*s+2*r**2+4*r*s+2*s**2],
            [2*r**2-r],
            [2*s**2-s],
            [4*r*s],
            [4*s-4*r*s-4*s**2],
            [4*r-4*r**2-4*r*s]
            ])

        dSdr = np.array([
            [-3+4*r+4*s], [4*r-1], [0], [4*s], [-4*s], [4-8*r-4*s]
            ])

        dSds = np.array([
            [-3+4*r+4*s], [0], [4*s-1], [4*r], [4-8*s-4*r], [-4*r]
            ])

        return S, dSdr, dSds

    def tri2edge(self):
        from ppp.PDE import sparse_matlab
        from scipy.sparse import triu, find
        i, j, k = self.T.T[:, None]
        A = sparse_matlab(i, k, -1, self.Np, self.Np) #1st edge is between (j, k)
        A += sparse_matlab(j, k, -1, self.Np, self.Np) #2nd
        A += sparse_matlab(i, j, -1, self.Np, self.Np) #3rd
        indices = (A + A.T).todense() < 0
        A = sp((self.Np, self.Np)).tolil()
        A[indices] = -1
        A = triu(A)
        r, c, v = find(A)
        v = np.linspace(0, len(v)-1, len(v))[:, None]
        A = sparse_matlab(r[None, :], c[None, :], v, self.Np, self.Np)
        A += A.T
        edges = np.zeros((self.Nt, 3), dtype=int)
        for k in range(self.Nt):
            edges[k] = np.array([
                    A[self.T[k, 1], self.T[k, 2]],
                    A[self.T[k, 0], self.T[k, 2]],
                    A[self.T[k, 0], self.T[k, 1]]
                ])

        return edges

    def changeP1toP2Mesh(self):
        edges = self.tri2edge() #get element edge numbers
        edges += self.Np #change edges to new nodes
        new_size = np.max(edges)+1
        i, j, k = self.T.T
        P = np.zeros((new_size, 2))
        P[:self.Np] = self.P

        e = edges[:, 0]
        P[e, 0] = .5*(self.P[j, 0]+self.P[k, 0]) #edge node coordinates
        P[e, 1] = .5*(self.P[j, 1]+self.P[k, 1])

        e = edges[:, 1]
        P[e, 0] = .5*(self.P[i, 0]+self.P[k, 0]) #edge node coordinates
        P[e, 1] = .5*(self.P[i, 1]+self.P[k, 1])

        e = edges[:, 2]
        P[e, 0] = .5*(self.P[i, 0]+self.P[j, 0]) #edge node coordinates
        P[e, 1] = .5*(self.P[i, 1]+self.P[j, 1])
        self.P = P
        self.Np0 = self.Np
        self.Np = new_size
        self.T = np.concatenate((self.T, edges), axis=1)

    def IsoStiffnessAssembler(self, func):
        rspts, qwgts = GaussPoints(4) #Quadrature Rule
        S = sp((self.Np, self.Np)).tolil()
        for i in range(self.Nt):
            loc2glb = self.T[i] #node numbers
            x, y = self.P[loc2glb].T #Node x- and y-coordinates
            SK = sp((6, 6)) #elements stiffness
            for q in range(len(qwgts)): #quadrature loop
                r = rspts[q, 0] #quadrature r-coordinate
                s = rspts[q, 1] #quadrature s-coordinate
                Sv, dSdx, dSdy, detJ = Isopmap(x, y, r, s, self.shapes)
                xc, yc = Sv[:, 0].dot(x), Sv[:, 0].dot(y)
                wxarea = qwgts[q] * detJ/2 #weight times area
                SK +=  (dSdx @ dSdx.T + \
                        dSdy @ dSdy.T) * wxarea * func(xc, yc) #Element Stiffness

            S[loc2glb[:, None], loc2glb[None,:]] += SK

        return S.tocsr()

    def IsoStrainXAssembler(self, func):
        rspts, qwgts = GaussPoints(4) #Quadrature Rule
        S = sp((self.Np, self.Np)).tolil()
        for i in range(self.Nt): #Loop through elements
            loc2glb = self.T[i] #node numbers
            x, y = self.P[loc2glb].T #Node x- and y-coordinates
            SK = sp((6, 6)) #Local Strain
            for q in range(len(qwgts)): #quadrature loop
                r = rspts[q, 0] #quadrature r-coordinate
                s = rspts[q, 1] #quadrature s-coordinate
                Sv, dSdx, dSdy, detJ = Isopmap(x, y, r, s, self.shapes)
                xc, yc = Sv[:, 0].dot(x), Sv[:, 0].dot(y)
                wxarea = qwgts[q] * detJ/2 #weight times area
                SK +=  (Sv @ dSdx.T) * wxarea * func(xc, yc) #Element Stiffness

            S[loc2glb[:, None], loc2glb[None,:]] += SK

        return S.tocsr()

    def IsoStrainYAssembler(self, func):
        rspts, qwgts = GaussPoints(4) #Quadrature Rule
        S = sp((self.Np, self.Np)).tolil()
        for i in range(self.Nt):
            loc2glb = self.T[i] #node numbers
            x, y = self.P[loc2glb].T #Node x- and y-coordinates
            SK = sp((6, 6)) #elements stiffness
            for q in range(len(qwgts)): #quadrature loop
                r = rspts[q, 0] #quadrature r-coordinate
                s = rspts[q, 1] #quadrature s-coordinate
                Sv, dSdx, dSdy, detJ = Isopmap(x, y, r, s, self.shapes)
                xc, yc = Sv[:, 0].dot(x), Sv[:, 0].dot(y)
                wxarea = qwgts[q] * detJ/2 #weight times area
                SK +=  (Sv @ dSdy.T) * wxarea * func(xc, yc) #Element Stiffness

            S[loc2glb[:, None], loc2glb[None,:]] += SK

        return S.tocsr()

    def IsoMassAssembler(self):
        rspts, qwgts = GaussPoints(4) #Quadrature Rule
        M = sp((self.Np, self.Np)).tolil()
        for i in range(self.Nt):
            loc2glb = self.T[i] #node numbers
            x, y = self.P[loc2glb].T #Node x- and y-coordinates
            MK = sp((6, 6)) #elements stiffness
            for q in range(len(qwgts)): #quadrature loop
                r = rspts[q, 0] #quadrature r-coordinate
                s = rspts[q, 1] #quadrature s-coordinate
                S, dSdx, dSdy, detJ = Isopmap(x, y, r, s, self.shapes)
                wxarea = qwgts[q] * detJ/2 #weight times area
                MK +=  (S @ S.T) * wxarea#Element Stiffness
            M[loc2glb[:, None], loc2glb[None,:]] += MK

        return M.tocsr()

    def show_mesh2D(self):
        import matplotlib.pyplot as pt
        from ppp.Plots import plot_setup
        fig, ax = plot_setup('$x$', '$y$')
        ax.triplot(self.P[:self.Np,0], self.P[:self.Np,1], self.T[:, :3])
        ax.plot(self.P[:,0], self.P[:,1], 'o')
        ax.plot(self.B[:, 0], self.B[:, 1], 'bx')
        self.draw_boundary(ax)
        pt.show()

    def draw_boundary(self, ax):
        for e in self.E:
            ax.plot(self.x[e], self.y[e], 'k-')


def GaussPoints(precision):
    """
    Gauss quadrature for the triangle.

    Parameters
    ----------
    precision : Int
        Describes the order of accuracy for the Gauss quadrature when integrating
        over the standard triangle.

    Raises
    ------
    ValueError
        When precision > 4 or non-integer, then a ValueError will be raised.

    Returns
    -------
    rspts : TYPE
        The 2D-reparameterisation from global to localisd standard
    qwgts : TYPE
        The integration weights (which should summate to 1).

    """
    if precision == 1:
        qwgts = np.array([1])
        rspts = np.ones(2)/3

    elif precision == 2:
        qwgts = np.ones(3)/3
        rspts = np.array([
            [1, 1], [4, 1], [1, 4]
            ])/6

    elif precision == 3:
        qwgts = np.array([
            -27/48, 25/48, 25/48, 25/48
            ])
        rspts = np.array([
            [1/3, 1/3], [.2, .2], [.6, .2], [.2, .6]
            ])

    elif precision == 4:
        qwgts = np.array([
            0.223381589678011, 0.223381589678011, 0.223381589678011,
            0.109951743655322, 0.109951743655322, 0.109951743655322
            ])
        rspts = np.array([
            [0.445948490915965, 0.445948490915965],
            [0.445948490915965, 0.108103018168070],
            [0.108103018168070, 0.445948490915965],
            [0.091576213509771, 0.091576213509771],
            [0.091576213509771, 0.816847572980459],
            [0.816847572980459, 0.091576213509771]
            ])

    else:
        raise ValueError("Quadrature precision too high on triangle")

    assert abs(np.sum(qwgts) - 1) < 1e-12 #Ensures that the weights do indeed sum to 1.
    return rspts, qwgts

def Isopmap(x, y, r, s, shapefcn):
    S, dSdr, dSds = shapefcn(r, s)
    j11, j12 = dSdr[:, 0].dot(x), dSdr[:, 0].dot(y)
    j21, j22 = dSds[:, 0].dot(x), dSds[:, 0].dot(y)
    detJ = j11*j22 - j12*j21
    dSdx = (j22*dSdr-j12*dSds)/detJ
    dSdy = (-j21*dSdr+j11*dSds)/detJ

    return S, dSdx, dSdy, detJ

if __name__ == '__main__':
    import dmsh

    geo1 = dmsh.Circle((0, 0), 1)
    geo2 = dmsh.Rectangle(0.5, 1.5, 0, 1)
    geo = dmsh.Union((geo1, geo2))
    P, T, E, B = dmsh.generate(geo, .5)
    fem = FEM_P2CG(P, T, E, B)
    fem.show_mesh2D()

    A = fem.IsoStiffnessAssembler(lambda x,y : 1)