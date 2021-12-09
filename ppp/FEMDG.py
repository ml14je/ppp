#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

The following class is based on (Hesthaven & Warburton, 2008)

Python 3.7
Created on Thu Feb 25 00:54:52 2021
"""
import numpy as np
from scipy.sparse import csr_matrix as sp
from scipy.linalg import solve, inv

class FEM(object):
    def __init__(self, P, T, N=1, BCs=None):
        """
        Setup script, building operators, grid, metric, and
        connectivity tables.

        Parameters
        ----------
        P : numpy.array
            Array describing points on triangular mesh
        T : numpy.array
            Array of element indices
        N : int, optional
            DG-order scheme. The default is 1.
        """

        #Mesh information
        self.P, self.T = P, T #Points, Element Triangles

        self.N = N #DG Order Scheme
        self.Nfp, self.Np, self.Nfaces = N+1, (N+1)*(N+2)//2, 3
        x, y = Nodes2D(self.N)
        self.r, self.s = xytors(x, y)

        self.NODETOL, self.eps = 1e-12, 1e-16

        #Build reference element matrices
        self.V = Vandermonde2D(self.N, self.r, self.s)
        self.Vinv = inv(self.V)
        self.mass_matrix = self.Vinv.T @ self.Vinv
        self.Dr, self.Ds = Dmatrices2D(self.N, self.r, self.s, self.V)
        self.base()
        self.BuildMaps2D()

        if BCs is not None:
            if type(BCs) == dict:
                self.CorrectBCTable(BCs)

            else:
                assert BCs.shape == (self.K, self.Nfaces)
                self.BCType = BCs
                self.BCs = {
                    'In': 1,
                    'Out': 2,
                    'Wall': 3,
                    'Far': 4,
                    'Cyl': 5,
                    'Dirichlet': 6,
                    'Neumann': 7,
                    'Slip': 8
                    }

            self.BuildBCMaps2D()

        else:
            self.BCType = None

        self.dtscale_compute()

        self.rk4a = np.array([
            0.0,
            -567301805773.0/1357537059087.0,
            -2404267990393.0/2016746695238.0,
            -3550918686646.0/2091501179385.0,
            -1275806237668.0/842570457699.0
            ])

        self.rk4b = np.array([
            1432997174477.0/9575080441755.0,
            5161836677717.0/13612068292357.0,
            1720146321549.0/2090206949498.0,
            3134564353537.0/4481467310338.0,
            2277821191437.0/14882151754819.0
            ])

        self.rk4c = np.array([
            0.0,
            1432997174477.0/9575080441755.0,
            2526269341429.0/6820363962896.0,
            2006345519317.0/3224310063776.0,
            2802321613138.0/2924317926251.0
            ])

    def base(self):
        #build coordinates of all the nodes
        self.va, self.vb, self.vc = self.T.T
        self.VX, self.VY = self.P.T
        self.Nv, self.K = len(self.VX), len(self.va)

        self.x = ((-(self.r+self.s)[:, None] @ self.VX[None, self.va] + \
                  (1+self.r)[:, None] @ self.VX[None, self.vb] + \
                      (1+self.s)[:, None] @ self.VX[None, self.vc])/2)
        self.y = ((-(self.r+self.s)[:, None] @ self.VY[None, self.va] + \
                  (1+self.r)[:, None] @ self.VY[None, self.vb] + \
                      (1+self.s)[:, None] @ self.VY[None, self.vc])/2)

        # Find all the nodes that lie on each edge
        fmask1 = np.where(abs(self.s+1) < self.NODETOL)
        fmask2 = np.where( abs(self.r+self.s) < self.NODETOL)
        fmask3 = np.where( abs(self.r+1) < self.NODETOL)
        self.Fmask = np.concatenate((fmask1, fmask2, fmask3)).T
        self.Fx, self.Fy = self.x[self.Fmask.flatten('F')], self.y[self.Fmask.flatten('F')]

        # Create surface integral terms
        self.Lift2D()

        # Build connectivity matrix
        self.Connect2D()

        # calculate geometric factors
        self.GeometricFactors2D()
        self.Normals2D()

        self.Vr, self.Vs = GradVandermonde2D(self.N, self.r, self.s)
        self.Drw = solve((self.V @ self.V.T), (self.V @ self.Vr.T).T).T
        self.Dsw = solve((self.V @ self.V.T), (self.V @ self.Vs.T).T).T

    def GeometricFactors2D(self):
        """
        Compute the metric elements for the local mappings
        of the elements
        """

        xr, xs= self.Dr @ self.x, self.Ds @ self.x
        yr, ys= self.Dr @ self.y, self.Ds @ self.y
        self.J = -xs*yr + xr*ys
        self.rx, self.sx = ys/self.J, -yr/self.J
        self.ry, self.sy = -xs/self.J, xr/self.J

    def Filter2D(self, Norder, Nc, Ns):
        """
        Initialize 2D filter matrix of order Ns and cutoff Nc
        """
        filterdiag = np.ones((Norder+1)*(Norder+2)//2)
        alpha = -np.log(self.eps)
        #build exponential filter
        sk = 0
        for i in range(Norder + 1):
            for j in range(Norder - i + 1):
                if i + j >= Nc:
                    filterdiag[sk] = np.exp(-alpha*((i+j - Nc)/(Norder-Nc))**Ns)
                sk += 1

        return self.V @ np.diag(filterdiag) @ self.invV

    def Lift2D(self):
        """
        Compute surface to volume lift term for DG formulation
        """

        Emat = np.zeros((self.Np, self.Nfaces*self.Nfp))

        # face 1
        faceR = self.r[self.Fmask[:, 0]]
        V1D = Vandermonde1D(self.N, faceR)
        massEdge1 = inv(V1D @ V1D.T)
        Emat[self.Fmask[:, 0], :self.Nfp] = massEdge1
        # face 2
        faceR = self.r[self.Fmask[:, 1]]
        V1D = Vandermonde1D(self.N, faceR);
        massEdge2 = inv(V1D @ V1D.T)
        Emat[self.Fmask[:, 1],self.Nfp:2*self.Nfp] = massEdge2
        # face 3
        faceS = self.s[self.Fmask[:, 2]]
        V1D = Vandermonde1D(self.N, faceS)
        massEdge3 = inv(V1D @ V1D.T)

        Emat[self.Fmask[:, 2], 2*self.Nfp:3*self.Nfp] = massEdge3
        # inv(mass matrix)*\I_n (L_i,L_j)_{edge_n}
        self.LIFT = self.V @ (self.V.T @ Emat)

    def Normals2D(self):
        """
        Compute outward pointing normals at elements faces and
        surface Jacobians
        """
        xr, xs= self.Dr @ self.x, self.Ds @ self.x
        yr, ys= self.Dr @ self.y, self.Ds @ self.y
        Fmask = self.Fmask.flatten('F')
        # Interpolate geometric factors to face nodes
        fxr, fxs = xr[Fmask, :], xs[Fmask, :]
        fyr, fys = yr[Fmask, :], ys[Fmask, :]

        # build normals
        nx, ny = np.zeros((3*self.Nfp, self.K)), np.zeros((3*self.Nfp, self.K))
        fid1 = np.linspace(0, self.Nfp-1, self.Nfp, dtype=int)
        fid2 = fid1 + self.Nfp
        fid3 = fid2 + self.Nfp
        # face 1
        nx[fid1], ny[fid1] = fyr[fid1], -fxr[fid1]

        # face 2
        nx[fid2], ny[fid2] = fys[fid2]-fyr[fid2], -fxs[fid2]+fxr[fid2]

        # face 3
        nx[fid3], ny[fid3] = -fys[fid3], fxs[fid3]

        # normalise
        self.sJ = np.sqrt(nx*nx + ny*ny)
        self.nx, self.ny = nx/self.sJ, ny/self.sJ
        self.Fscale  = self.sJ/self.J[Fmask]

    def Connect2D(self):
        """
        Build global connectivity arrays for grid based on
        standard Triangular-Element (T) input array from grid generator.
        """
        Nfaces = 3
        Nv = np.max(self.T) + 1

        # Create face to node connectivity matrix
        TotalFaces = Nfaces * self.K

        # List of local face to local vertex connections
        vn = np.array([
            [0, 1], [1, 2], [0, 2]
            ])

        # Build global face-to-node sparse array
        SpFToV = sp((TotalFaces, Nv)).tolil()

        sk = 0
        for k in range(self.K):
            for face in range(Nfaces):
                inds = self.T[k, vn[face]]
                SpFToV[sk*np.ones(len(inds)), inds] = 1
                sk += 1

        from scipy.sparse import eye, find

        # Build global face to global face sparse array
        SpFToF = SpFToV @ SpFToV.T - 2*eye(TotalFaces)

        # Find complete face to face connections
        faces1, faces2, b = find(SpFToF==2)

        # Convert face global number to element and face numbers
        element1 = np.floor(faces1/Nfaces).astype(int)
        face1 = (faces1 % Nfaces).astype(int)
        element2 = np.floor(faces2/Nfaces).astype(int)
        face2 = (faces2 % Nfaces).astype(int)

        # Rearrange into Nelements x Nfaces sized arrays
        EToE = np.linspace(0, self.K-1, self.K)[:, None] @ np.ones((1, Nfaces))
        EToF = np.ones((self.K, 1)) @ np.linspace(0, Nfaces-1, Nfaces)[None, :]

        EToE[element1, face1] = element2
        EToF[element1, face1] = face2
        self.EToE, self.EToF = EToE.astype(int), EToF.astype(int)

    def BuildMaps2D(self):
        """
        Connectivity and boundary tables in the K #
        of Np elements
        """
        # number volume nodes consecutively
        nodeids = np.linspace(0, self.K*self.Np-1,
                              self.K*self.Np, dtype=int).reshape((self.K, self.Np)).T
        vmapM = np.zeros((self.Nfp, self.Nfaces, self.K), dtype=int)
        vmapP = np.zeros((self.Nfp, self.Nfaces, self.K), dtype=int)
        n = self.K*self.Nfp*self.Nfaces
        mapM = np.linspace(0, n-1, n, dtype=int)[:, None]
        mapP = np.copy(mapM).reshape((self.Nfp, self.Nfaces, self.K))
        for k1 in range(self.K):
            for f1 in range(self.Nfaces):
                vmapM[:,f1,k1] = nodeids[self.Fmask[:, f1], k1]

        one = np.ones((1, self.Nfp))
        for k1 in range(self.K):
            for f1 in range(self.Nfaces):
                # find neighbor
                k2, f2 = self.EToE[k1,f1], self.EToF[k1,f1]
                # reference length of edge
                v1, v2= self.T[k1,f1], self.T[k1, (f1+1)%self.Nfaces]
                refd = np.sqrt( (self.VX[v1]-self.VX[v2])**2 + \
                                (self.VY[v1]-self.VY[v2])**2)

                # find find volume node numbers of left and right nodes
                vidM, vidP = vmapM[:,f1,k1], vmapM[:,f2,k2]
                x1, y1 = self.x.flatten('F')[vidM], self.y.flatten('F')[vidM]
                x2, y2 = self.x.flatten('F')[vidP], self.y.flatten('F')[vidP]
                x1, y1 = x1[:, None] @ one, y1[:, None] @ one
                x2, y2 = x2[:, None] @ one, y2[:, None] @ one

                # Compute distance matrix
                D = (x1 -x2.T)**2 + (y1-y2.T)**2
                idM, idP = np.where(np.sqrt(abs(D))<self.NODETOL*refd)
                vmapP[idM, f1, k1] = vidP[idP]
                mapP[idM, f1, k1] = idP + f2 * self.Nfp + \
                                    k2 * self.Nfaces * self.Nfp

        # Reshape vmapM and vmapP to be vectors and create boundary node list
        self.vmapP = vmapP.flatten('F')
        self.vmapM = vmapM.flatten('F')
        self.mapP, self.mapM = mapP.flatten('F'), mapM.flatten('F')
        self.mapB = np.where(self.vmapP==self.vmapM)[0]
        self.vmapB = self.vmapM[self.mapB]

    def BuildBCMaps2D(self):
        """
        Build specialized nodal maps for various types of
        boundary conditions, specified in BCType.
        """

        bct = self.BCType.T
        bnodes = np.ones((self.Nfp, 1)) @ (bct.flatten('F')[None, :])
        bnodes = bnodes.flatten('F')
        self.maps, self.vmaps = {}, {}

        for i, BC in enumerate(self.BCs.keys()):
            self.maps[BC] = np.where(bnodes == self.BCs[BC])[0]
            self.vmaps[BC] = self.vmapM[self.maps[BC]]

    def CorrectBCTable(self, BCs):
        self.BCs = dict(zip(BCs.keys(), np.arange(len(BCs.keys()))+1))

        self.BCType = np.zeros(self.EToE.shape, dtype=int)
        VNUM = np.array([[1, 2], [2, 3], [3, 1]]) - 1

        for BC in self.BCs.keys():
            BCcode = self.BCs[BC]
            BC_map = BCs[BC]

            for k in range(self.K):
                #Test for each edge
                for l in range(self.Nfaces):
                    m, n = self.T[k, VNUM[l, :2]]
                    # if both points are on the boundary then it is a boundary
                    # point!
                    if (m in BC_map) and (n in BC_map):
                        self.BCType[k, l] = BCcode

    def dtscale_compute(self):
        """
        Compute inscribed circle diameter as characteristic
        for grid to choose timestep
        """
        vmask1 = np.where(abs( self.s+ self.r+2) <  self.NODETOL)[0]
        vmask2 = np.where( abs(self.r-1) < self.NODETOL)[0]
        vmask3 = np.where( abs(self.s-1) < self.NODETOL)[0]
        vmask = np.concatenate([vmask1, vmask2, vmask3])
        vx, vy = self.x[vmask], self.y[vmask]

        # Compute semi-perimeter and area
        len1 = np.sqrt((vx[0]-vx[1])**2 + (vy[0]-vy[1])**2)
        len2 = np.sqrt((vx[1]-vx[2])**2 + (vy[1]-vy[2])**2)
        len3 = np.sqrt((vx[2]-vx[0])**2 + (vy[2]-vy[0])**2)
        sper = (len1 + len2 + len3)/2
        Area = np.sqrt(sper * (sper-len1) * (sper-len2) * (sper-len3))
        # Compute scale using radius of inscribed circle
        self.dtscale = Area/sper

    def Curl2D(self, ux, uy, uz):
        """
        Compute 2D curl-operator in (x,y) plane of vector (ux, uy, uz)
        """
        n = len(ux)
        uxr, uxs = self.Dr @ ux, self.Ds @ ux
        uyr, uys = self.Dr @ uy, self.Ds @ uy

        vz = self.rx * uyr + self.sx * uys - self.ry * uxr - self.sy * uxs

        vx, vy = np.zeros((n, 1)), np.zeros((n, 1))
        if not np.all(uz) == 0:
            uzr, uzs = self.Dr @ uz, self.Ds @ uz
            vx = self.ry * uzr + self.sy * uzs
            vy = -self.rx * uzr - self.sx * uzs

        return vx, vy, vz

    def Div2D(self, u, v):
        """
        Compute the 2D divergence of the vectorfield (u,v)
        """

        ur, us, vr, vs = self.Dr @ u, self.Ds @ u, self.Dr @ v, self.Ds @ v

        divu = self.rx * ur + self.sx * us + self.ry * vr + self.sy * vs

        return divu

    def Grad2D(self, u):
        """
        Compute 2D gradient field of scalar u
        """
        ur, us = self.Dr @ u, self.Ds @ u
        ux, uy = self.rx * ur + self.sx * us, self.ry * ur + self.sy * us

        return ux, uy

    def plot_mesh(self):
        oFx = (self.Fx.T.reshape(self.Nfaces*self.K, self.Nfp)).T
        oFy = (self.Fy.T.reshape(self.Nfaces*self.K, self.Nfp)).T

        from ppp.Plots import plot_setup
        import matplotlib.pyplot as pt

        fig, ax= plot_setup('$x$', '$y$', title='Mesh')
        ax.plot(oFx, oFy, 'k-')

        pt.show()

    def PlotField2D(self, Nout, xin, yin, uin, title='Field', name=None):
        TRI, xout, yout, uout, interp = self.FormatData2D(Nout, xin, yin, uin)

        import plotly.figure_factory as ff
        fig = ff.create_trisurf(x=xout.flatten('F'), y=yout.flatten('F'),
                                z=uout.flatten('F'), simplices=TRI,
                                title=title)
        if name is None:
            fig.show()

        else:
            fig.write_html(name+'.html', auto_open=True, include_mathjax='cdn')

        return TRI, xout, yout, uout, interp

    def FormatData2D(self, Nout, xin, yin, uin):
        Npout = (Nout + 1)*(Nout+2)//2
        rout, sout = np.zeros((Npout, 1)), np.zeros((Npout, 1))
        sk = 0
        counter = np.zeros((Nout+1, Nout+1))
        for n in range(Nout+1):
            for m in range(Nout+1-n):
                rout[sk], sout[sk] = -1 + 2*m/Nout, -1 + 2*n/Nout
                counter[n, m] = sk
                sk += 1

        #Build matrix to interpolate field to equally-spaced nodes
        interp = self.InterpMatrix2D(rout, sout)
        from scipy.sparse import block_diag
        interp = block_diag([interp]*self.K)
        tri = []

        for n in range(Nout):
            for m in range(Nout-n):
                v1, v2 = counter[n, m], counter[n, m+1]
                v3, v4 = counter[n+1, m], counter[n+1, m+1]

                if v4:
                    tri.append([v1, v2, v3])
                    tri.append([v2, v4, v3])

                else:
                    tri.append([v1, v2, v3])

        tri = np.array(tri)
        n = tri.shape[0]

        # build triangulation for all equally spaced nodes on all elements
        TRI = np.zeros((self.K*n, 3), dtype=int)
        for k in range(self.K):
            TRI[k*n:n*(k+1)] = tri+k*Npout

        xout = interp @ xin
        yout = interp @ yin

        if len(uin.shape) == 1:
            uout = (interp @ uin[:, None]).T

        elif len(uin.shape) == 2:
            uout = (interp @ uin.T).T

        elif len(uin.shape) == 3:
            uout = np.zeros((uin.shape[0], interp.shape[0], uin.shape[2]))
            for i, u in enumerate(uin):
                uout[i] = (interp @ u)

        elif len(uin.shape) == 4:
            uout = np.zeros((uin.shape[0], uin.shape[1], interp.shape[0], uin.shape[3]))

            for j in range(uin.shape[0]):
                uvals = uin[j]
                for i, u in enumerate(uvals):
                    uout[j, i] = (interp @ u)

        else:
            raise ValueError('Invalid u_in shape size')

        return TRI, xout, yout, uout, interp

    def PlotContour2D(self, Nout, xin, yin, uin, levels, title='Field', name=None):
        """
        Generic routine to plot contours for triangulated data
        """
        tri, x, y, u, interp = self.FormatData2D(Nout, xin, yin, uin)
        Nlevels = len(levels)
        v1, v2, v3 = tri.T
        u1, u2, u3 = u.flatten('F')[v1], u.flatten('F')[v2], u.flatten('F')[v3]
        x1, x2, x3 = x.flatten('F')[v1], x.flatten('F')[v2], x.flatten('F')[v3]
        y1, y2, y3 = y.flatten('F')[v1], y.flatten('F')[v2], y.flatten('F')[v3]

        allx, ally = np.empty((2,1)), np.empty((2,1))
        for n in range(Nlevels):
            lev = levels[n]
            flag1 = (np.maximum(u1, u2) >= lev) & (np.minimum(u1, u2) <= lev)   # edge 1
            flag2 = (np.maximum(u3, u2) >= lev) & (np.minimum(u3, u2) <= lev)   # edge 2
            flag3 = (np.maximum(u1, u3) >= lev) & (np.minimum(u1, u3) <= lev)   # edge 3

            c1, c2, c3 = (lev-u1)/(u2-u1), (lev-u2)/(u3-u2), (lev-u1)/(u3-u1)
            xc1, yc1 = (1-c1)*x1 + c1*x2, (1-c1)*y1 + c1*y2
            xc2, yc2 = (1-c2)*x2 + c2*x3, (1-c2)*y2 + c2*y3
            xc3, yc3 = (1-c3)*x1 + c3*x3, (1-c3)*y1 + c3*y3

            ids = flag1 & flag2
            if np.any(ids):
                allx = np.append(allx, np.concatenate([xc1[None, ids], xc2[None, ids]]), axis=1)
                ally = np.append(ally, np.concatenate([yc1[None, ids], yc2[None, ids]]), axis=1)

            ids = flag2 & flag3
            if np.any(ids):
                allx = np.append(allx, np.concatenate([xc2[None, ids], xc3[None, ids]]), axis=1)
                ally = np.append(ally, np.concatenate([yc2[None, ids], yc3[None, ids]]), axis=1)

            ids = flag1 & flag3
            if np.any(ids):
                allx = np.append(allx, np.concatenate([xc1[None, ids], xc3[None, ids]]), axis=1)
                ally = np.append(ally, np.concatenate([yc1[None, ids], yc3[None, ids]]), axis=1)

        allx, ally = allx[:, 1:], ally[:, 1:]

        from ppp.Plots import plot_setup
        import matplotlib.pyplot as pt
        fig, ax = plot_setup('$x$', '$y$')
        ax.plot(allx, ally, color='r')
        pt.show()

    def InterpMatrix2D(self, rout, sout):
        """Compute local elemental interpolation matrix
        """

        # compute Vandermonde at (rout,sout)
        Vout = Vandermonde2D(self.N, rout.T[0], sout.T[0]);

        # build interpolation matrix
        return Vout @ self.Vinv

    def PlotDomain2D(self):
        from ppp.Plots import plot_setup
        fig, ax = plot_setup('$x$', '$y$', title='Domain Boundary Map')

        if self.BCType is None:
            pass

        else:
            BCType = self.BCType.flatten('F')
            Ndict = dict()
            for BC in self.BCs.keys():
                Ndict[BC] = len(np.where(BCType==self.BCs[BC])[0])

            linestyles = ['-', '--', ':', '-.']
            sk, types = 0, []

            for BC in self.BCs.keys():
                if Ndict[BC]:
                    ax.plot([0,0],[0,0], 'k'+ linestyles[(self.BCs[BC]-1)%4], label=BC)
                    types.append(BC)
                    sk+=1

            ax.legend(fontsize=16, loc=4)

            Fx, Fy = self.Fx.flatten('F'), self.Fy.flatten('F')

            for k in range(self.K): #Element
                for f in range(self.Nfaces): #Face
                    bc = self.BCType[k, f]
                    if bc != 0:
                        ids = k*self.Nfp*self.Nfaces+f*self.Nfp+np.arange(self.Nfp)

                        ax.plot(Fx[ids], Fy[ids], 'k'+ linestyles[(bc-1)%4])

        return fig, ax

    def Hrefine2D(self, refineflag):
        """
        apply non-conforming refinement to elements labelled
        in refineflag
        """

        # Count vertices
        Nv = len(self.VX.flatten('F'))
        # Find and count elements to be refined
        ref = np.sort(np.where(refineflag==True)[0])
        Nrefine = len(ref)
        # Extract vertex numbers of elements to refine
        v1, v2, v3 = self.T[ref, 0], self.T[ref, 1], self.T[ref, 2]

        # Uniquely number all face centers
        v4 = np.maximum(0 + self.Nfaces*np.arange(self.K, dtype=int), self.EToF[:, 0] + self.Nfaces * (self.EToE[:, 0] )).astype(int)
        v5 = np.maximum(1 + self.Nfaces*np.arange(self.K, dtype=int), self.EToF[:, 1] + self.Nfaces * (self.EToE[:, 1] )).astype(int)
        v6 = np.maximum(2 + self.Nfaces*np.arange(self.K, dtype=int), self.EToF[:, 2] + self.Nfaces * (self.EToE[:, 2] )).astype(int)

        # Extract face center vertices for elements to refine
        v4, v5, v6 = v4[ref],  v5[ref], v6[ref]

        # Renumber face centers contiguously from Nv+1
        ids = np.unique(np.concatenate([v4, v5, v6]))
        newids = np.zeros(np.max(ids)+1)

        newids[ids] = np.arange(len(ids+1), dtype=int)
        v4, v5, v6 = (Nv+newids[v4]).astype(int), (Nv+newids[v5]).astype(int), (Nv+newids[v6]).astype(int)

        # Replace original triangle with triangle connecting edge centers
        self.T[ref, 0], self.T[ref, 1], self.T[ref, 2] = v4, v5, v6

        # Add extra triangles to EToV
        self.T = np.concatenate((self.T, np.zeros((3*Nrefine, 3), dtype=int)), axis=0)
        self.T[self.K:self.K+3*Nrefine, 0] = np.concatenate((v1, v2, v3)) #first vertices of new elements
        self.T[self.K:self.K+3*Nrefine, 1] = np.concatenate((v4, v5, v6)) # second vertices of new elements
        self.T[self.K:self.K+3*Nrefine, 2] = np.concatenate((v6, v4, v5)) # third vertices of new elements

        # Create boundary condition type for refined elements
        bcsave = np.copy(self.BCType[ref])
        self.BCType[ref] = 0 # no internal faces
        self.BCType =  np.concatenate((self.BCType, np.zeros((3*Nrefine, 3), dtype=int)), axis=0)
        self.BCType[self.K:self.K+Nrefine, 0] = bcsave[:, 0]
        self.BCType[self.K:self.K+Nrefine, 2] = bcsave[:, 2]

        self.BCType[self.K+Nrefine:self.K+2*Nrefine, 0] = bcsave[:, 1]
        self.BCType[self.K+Nrefine:self.K+2*Nrefine, 2] = bcsave[:, 0]

        self.BCType[self.K+2*Nrefine:self.K+3*Nrefine, 0] = bcsave[:, 2]
        self.BCType[self.K+2*Nrefine:self.K+3*Nrefine, 2] = bcsave[:, 1]

        # Find vertex locations of elements to be refined
        x1 = self.VX[v1]; x2 = self.VX[v2]; x3 = self.VX[v3]
        y1 = self.VY[v1]; y2 = self.VY[v2]; y3 = self.VY[v3]

        self.VX = np.concatenate((self.VX, np.zeros(np.max(v6) - self.VX.shape[0] + 1)), axis=0)
        self.VY = np.concatenate((self.VY, np.zeros(np.max(v6) - self.VY.shape[0] + 1)), axis=0)

        # Add coordinates for refined edge centers
        self.VX[v4] = .5*(x1+x2); self.VX[v5] = .5*(x2+x3); self.VX[v6] = .5*(x3+x1)
        self.VY[v4] = .5*(y1+y2); self.VY[v5] = .5*(y2+y3); self.VY[v6] = .5*(y3+y1)

        self.P = np.concatenate((self.VX[:, None], self.VY[:, None]), axis=1)

        # Increase element count
        self.K += 3 * Nrefine
        Nv_old = self.Nv
        self.base()

    def BuildHNonCon2D(self, NGauss, tol):
        from ppp.Jacobi import JacobiGQ
        # Build Gauss nodes
        gz, gw = JacobiGQ(0, 0, NGauss-1)

        # Find location of vertices of boundary faces
        vx1, vx2 = self.VX[self.T[:, [0, 1, 2]]], self.VX[self.T[:, [1, 2, 0]]]
        vy1, vy2 = self.VY[self.T[:, [0, 1, 2]]], self.VY[self.T[:, [1, 2, 0]]]
        ints = (np.arange(self.K, dtype=int)[:, None] @ np.ones((1, self.Nfaces)))

        idB = np.where(self.EToE.flatten('F')==ints.flatten('F'))[0]

        x1, y1 = vx1.flatten('F')[idB], vy1.flatten('F')[idB]
        x2, y2 = vx2.flatten('F')[idB], vy2.flatten('F')[idB]

        # Find those element-faces that are on boundary faces
        elmtsB = np.where(self.EToE.flatten('F')==ints.flatten('F'))[0] % self.K
        facesB = np.where(self.EToE.flatten('F')==ints.flatten('F'))[0]//self.K

        Nbc = len(elmtsB)

        # For each boundary face
        self.neighbours = []
        for b1 in range(Nbc):
            # Find element and face of this boundary face
            k1, f1 = elmtsB[b1], facesB[b1]
            # Find end coordinates of b1’th boundary face
            x11, y11 = x1[b1], y1[b1]
            x12, y12 = x2[b1], y2[b1]

            # Compute areas, lengths and face coordinates used in
            # intersection tests comparing b1’th boundary face with
            # all boundary faces
            area1 = abs((x12-x11)*(y1-y11) - (y12-y11)*(x1-x11)) #scale
            area2 = abs((x12-x11)*(y2-y11) - (y12-y11)*(x2-x11))

            L = (x12-x11)**2 + (y12-y11)**2
            r21 = ((2*x1-x11-x12)*(x12-x11) + (2*y1-y11-y12)*(y12-y11))/L
            r22 = ((2*x2-x11-x12)*(x12-x11) + (2*y2-y11-y12)*(y12-y11))/L

            r1, r2 = np.maximum(-1, np.minimum(r21, r22)), np.minimum(1, np.maximum(r21, r22))


            # Compute flag for overlap of b1 face with all other
            # boundary faces
            bool1, bool2 = ((r1<= -1) & (r2<= -1)).astype(int), ((r1>=1) & (r2>=1)).astype(int)
            bool3 = (r2-r1 < tol).astype(int)
            flag = area1 + area2 + bool1 + bool2 + bool3

            # Find other faces with partial matches
            matches = np.setxor1d(np.where(flag < tol)[0], b1)
            Nmatches = len(matches)
            if Nmatches > 0:
                # Find matches
                r1, r2 = r1[None, matches], r2[None, matches]
                # Find end points of boundary-boundary intersections
                coords11, coords12 = np.array([x11, y11])[:, None], np.array([x12, y12])[:, None]
                xy11 = .5 * (coords11*(1-r1) + coords12*(1+r1))
                xy12 = .5 * (coords11*(1-r2) + coords12*(1+r2))

                # For each face-face match
                for n in range(Nmatches):
                    # Store which elements intersect
                    k2, f2 = elmtsB[matches[n]], facesB[matches[n]]
                    self.neighbours.append(Neighbor(k1, k2, f1, f2))

                    # Build physical Gauss nodes on face fragment
                    xg = .5* ((1 - gz) * xy11[0, n] + (1 + gz) * xy12[0, n])
                    yg = .5* ((1 - gz) * xy11[1, n] + (1 + gz) * xy12[1, n])

                    # Find local coordinates of Gauss nodes
                    rg1, sg1 = self.FindLocalCoords2D(k1, xg, yg)
                    rg2, sg2 = self.FindLocalCoords2D(k2, xg, yg)

                    # Build interpolation matrices for volume nodes ->Gauss nodes
                    gVM, gVP = self.InterpMatrix2D(rg1, sg1), self.InterpMatrix2D(rg2,sg2)
                    self.neighbours[-1].gVM = gVM
                    self.neighbours[-1].gVP = gVP

                    # Find face normal
                    self.neighbours[-1].nx = self.nx[f1*self.Nfp, k1]
                    self.neighbours[-1].ny = self.ny[f1*self.Nfp, k1]
                    # Build partial face data lift operator

                    # Compute weights for lifting
                    partsJ = np.sqrt( (xy11[0, n] - xy12[0, n])**2 + (xy11[1, n] - xy12[1, n])**2)/2
                    dgw = gw * partsJ/self.J[0, k1]

                    # Build matrix to lift Gauss data to volume data
                    self.neighbours[-1].lift = self.V @ self.V.T @ (gVM.T) @ np.diag(dgw[:, 0])

        self.Nneighbours = len(self.neighbours)

    def NonConformingHFluxCorrection(self, Lin_op=None, Flux=None, vec=None):
        """


        Parameters
        ----------
        M : matrix
            Linear matrix operator
        F : maxtrix
            Linear flux operator

        Returns
        -------
        Correct matrix operator which corrects the flux due to the non-conforming mesh
        """
        from scipy.sparse import diags
        # print(vec.shape)
        self.C = sp((self.K, self.K))
        N = self.K * self.Np
        Nv = 3 if vec is None else vec.shape[0]//N
        Im, Ip = sp((self.Nneighbours, self.K*self.Np)).tolil(), sp((self.Nneighbours, self.K*self.Np)).tolil()
        # Im, Ip = sp((N, N)).tolil(), sp((N, N)).tolil()

        M_vals, P_vals = np.empty(self.Nneighbours, dtype=int), np.empty(self.Nneighbours, dtype=int)
        # print(self.Nneighbours)
        gVMs, gVPs = np.empty((self.Nneighbours, self.N+1, self.Np)), np.empty((self.Nneighbours, self.N+1, self.Np))
        gVM, gVP = sp((self.Nfp * self.Np, self.Nneighbours)).tolil(), sp((self.Nfp * self.Np, self.Nneighbours)).tolil()
        vec = vec.reshape((self.K, self.Np))
        # print(self.Nfp * self.Np)
        for i, n in enumerate(self.neighbours):
            M_vals[i], P_vals[i] = n.elmtM, n.elmtP
            gVMs[i], gVPs[i] = n.gVM, n.gVP
            Im[i, n.elmtM:n.elmtM+self.Np], Ip[i, n.elmtM:n.elmtM+self.Np] = 1, 1
            # print(n.gVM.flatten('F').shape)
            gVM[:, i], gVP[:, i] = n.gVM.flatten('F'), n.gVP.flatten('F')
            # C = .5 *

        print(np.unique(M_vals), self.Nneighbours)

        # print(Im.todense())
        print(self.Nneighbours, self.K * self.Np)
        print(Im.shape, vec.shape)
        print((gVM @ Im @ vec).shape)
        raise ValueError










        # M_vals = np.array([n.elmtM for n in self.neighbours])
        # P_vals = np.array([n.elmtP for n in self.neighbours])
        # gVMs, gVPs = np.array([n.gVM for n in self.nighbours])
        # print(M_vals, P_vals)

        Im[M_vals], Ip[P_vals] = 1, 1
        Im = np.repeat(Im, self.Np)
        # print(Im)


        Im, Ip = diags(Im), diags(Ip)

        print(Im @ vec)
        raise ValueError
        # Only apply PEC boundary conditions at wall boundaries
        savemapB, savevmapB = np.copy(self.mapB), np.copy(self.vmapB)
        self.mapB, self.vmapB = np.copy(self.maps['Wall']), np.copy(self.vmaps['Wall'])

        # Evaluate right hand side
        # vec1 = Lin_op @ vec

        # Restore original boundary node lists
        self.mapB, self.vmapB = savemapB, savevmapB

        # Correct lifted fluxes at each non-conforming face fragment
        Nnoncon = len(self.neighbours)
        for n in range(Nnoncon):
            neigh = self.neighbours[n]

            # Extract information about this non-conforming face fragment
            k1, gVM = neigh.elmtM, neigh.gVM
            k2, gVP = neigh.elmtP, neigh.gVP
            lnx, lny = neigh.nx, neigh.ny
            print(k1, lnx, lny, gVM.shape, self.Np)
            raise ValueError

        #     # Compute difference of traces at Gauss nodes on face fragment
        #     from scipy.sparse import block_diag
        #     ld = block_diag(3 * [gVM @ Hx[:, k1] - gVP @ Hx[:, k2]])

        #     # Compute flux terms at Gauss nodes on face fragment
        #     # lndotdH =  lnx.*ldHx+lny.*ldHy

        #     fluxHx =  lny * ldEz + lndotdH * lnx-ldHx
        #     fluxHy = -lnx * ldEz + lndotdH * lny-ldHy
        #     fluxEz = -lnx * ldHy + lny * ldHx - ldEz

        #     # Lift fluxes for non-conforming face fragments and update residuals
        #     lift = neigh.lift;
        #     rhsHx[:, k1] = rhsHx[:, k1] + lift @ fluxHx/2
        #     rhsHy[:, k1] = rhsHy[:, k1] + lift @ fluxHy/2
        #     rhsEz[:, k1] = rhsEz[:, k1] + lift @ fluxEz/2


    def FindLocalCoords2D(self, k, xout, yout):
        """
        find local (r,s) coordinates in the k'th element of given coordinates
        [only works for straight sided triangles]
        """
        v1, v2, v3  = self.T[k, 0], self.T[k, 1], self.T[k, 2]
        xy1 = np.array([self.VX[v1], self.VY[v1]])
        xy2 = np.array([self.VX[v2], self.VY[v2]])
        xy3 = np.array([self.VX[v3], self.VY[v3]])

        A = np.array([xy2 - xy1, xy3 - xy1]).T
        # print('A', A)

        rOUT, sOUT = np.zeros(xout.shape), np.zeros(xout.shape)

        for i in range(len(xout.flatten('F'))):
            rhs = 2 * np.concatenate((xout[i], yout[i])) - xy2 - xy3

            tmp = solve(A, rhs[:, None])[:, 0]
            rOUT[i], sOUT[i] = tmp[0], tmp[1]

        return rOUT, sOUT

class Neighbor(object):
    def __init__(self, elmtM, elmtP, faceM, faceP):
        self.elmtM, self.elmtP = elmtM, elmtP
        self.faceM, self.faceP = faceM, faceP

def rstoab(r, s):
    """
    Transfer from (r,s) -> (a,b) coordinates of a triangle
    """
    Np = len(r)
    a = np.zeros(Np)
    for n in range(Np):
        if s[n] != 1:
            a[n] = 2*(1+r[n])/(1-s[n])-1
        else:
            a[n] = -1

    b = s
    return a, b

def xytors(x, y):
    """
    From (x,y) in equilateral triangle to (r,s) coordinates
    in standard triangle
    """
    L1 = (np.sqrt(3)*y+1)/3
    L2 = (-3*x - np.sqrt(3)*y + 2)/6
    L3 = (3*x - np.sqrt(3)*y + 2.0)/6
    r, s = -L2 + L3 - L1, -L2 - L3 + L1

    return r, s

def Warpfactor(N, rout):
    """
    Compute scaled warp function at order N based on
    rout interpolation nodes
    """
    #Compute LGL and equidistant node distribution
    from ppp.Jacobi import JacobiGL, JacobiP
    LGLr = JacobiGL(0, 0, N)
    req = np.linspace(-1, 1, N+1)
    Veq = Vandermonde1D(N, req) #Compute V based on req

    Nr = len(rout)
    Pmat = np.zeros((N+1, Nr))
    #Evaluate Lagrange polynomial at rout
    for i in range(N+1):
        vec = JacobiP(rout[:, None], 0, 0, i)
        Pmat[i] = vec.T


    Lmat = solve(Veq.T, Pmat)

    #Compute warp factor
    warp = Lmat.T @ (LGLr - req[:, None])

    #Scale factor
    zerof = (abs(rout)<1.0-1.0e-10).astype(int)
    sf = 1 - (zerof * rout)**2
    return warp/sf[:, None] + warp*(zerof[:, None]-1)

def Nodes2D(N):
    alpopt = np.array([0, 0, 1.4152, 0.1001, 0.2751, 0.98, 1.0999, 1.2832, 1.3648,
              1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258])

    alpha = alpopt[N-1] if N < 16 else 5/3

    Np = (N+1)*(N+2)//2

    L1, L2, L3 = np.zeros(Np), np.zeros(Np), np.zeros(Np)
    sk = 0
    for n in range(N+1):
        for m in range(N+2-n-1):
            L1[sk], L3[sk] = n/N, m/N
            sk += 1

    L2 = 1 - L1 - L3
    x, y = -L2 + L3, (-L2 - L3 + 2 * L1)/np.sqrt(3)
    #Compute blending function at each node for each edge
    blend1, blend2, blend3 = 4 * L2 * L3, 4 * L1 * L3, 4 * L1 * L2
    #Amount of warp for each node, for each edge
    warpf1, warpf2 = Warpfactor(N, L3 - L2)[:, 0], Warpfactor(N, L1 - L3)[:, 0]
    warpf3 = Warpfactor(N, L2 - L1)[:, 0]
    #Combine blend & warp
    warp1 = blend1 * warpf1 * (1 + (alpha * L1)**2)
    warp2 = blend2 * warpf2 * (1 + (alpha * L2)**2)
    warp3 = blend3 * warpf3 * (1 + (alpha * L3)**2)

    #Accumulate deformations associated with each edge
    x += 1*warp1 + np.cos(2*np.pi/3)*warp2 + np.cos(4*np.pi/3)*warp3
    y += 0*warp1 + np.sin(2*np.pi/3)*warp2 + np.sin(4*np.pi/3)*warp3

    return x, y

def Vandermonde1D(N, r):
    """
    Initialize the 1D Vandermonde Matrix, V_{ij} = phi_j(r_i)
    """
    from ppp.Jacobi import JacobiP
    V1D = np.zeros((len(r), N+1))
    for j in range(N+1):
        V1D[:, j] = JacobiP(r[:, None], 0, 0, j)

    return V1D

def Vandermonde2D(N, r, s):
    """
    Initialize the 1D Vandermonde Matrix, V_{ij} = phi_j(r_i, s_i)
    """
    from ppp.Jacobi import Simplex2DP
    V2D = np.zeros((len(r), (N+1)*(N+2)//2))
    a, b = rstoab(r, s)
    sk = 0
    for i in range(N+1):
        for j in range(N-i+1):
            V2D[:, sk] = Simplex2DP(a, b, i, j)
            sk += 1

    return V2D

def Dmatrices2D(N, r, s, V):
    """
    Initialize the (r,s) differentiation matrices
    on the simplex, evaluated at (r,s) at order N
    """
    Vr, Vs = GradVandermonde2D(N, r, s)

    Dr, Ds = solve(V.T, Vr.T).T, solve(V.T, Vs.T).T

    return Dr, Ds

def GradVandermonde2D(N, r, s):
    """
    Initialize the gradient of the modal basis (i,j)
    at (r,s) at order N
    """
    from ppp.Jacobi import GradSimplex2DP
    V2Dr, V2Ds = np.zeros((len(r), (N+1)*(N+2)//2)), np.zeros((len(r), (N+1)*(N+2)//2))
    a, b = rstoab(r, s)

    sk = 0
    for i in range(N+1):
        for j in range(N-i+1):
            V2Dr[:, sk], V2Ds[:, sk] = GradSimplex2DP(a, b, i, j)
            sk += 1

    return V2Dr, V2Ds

def multi_boundary(FileName, order):
    from ppp.MeshReader2D import MeshReaderGambitBC
    Nv, VX, VY, K, EToV, BCType = MeshReaderGambitBC(FileName)
    T = EToV
    P, T = np.concatenate((VX[:, None], VY[:, None]), axis=1), EToV

    X, Y = P.T
    inflow_inds = np.where(X==-.5)[0]
    outflow_inds = np.where(X==1)[0]
    wall_inds = np.where(Y==-.5)[0]
    open_inds = np.where(Y==1.5)[0]

    BC_maps, BC_Types = [inflow_inds, outflow_inds, wall_inds, open_inds],\
                    ['Inflow', 'Outflow', 'Wall', 'Open']
    BCs = dict(zip(BC_Types, BC_maps))

    fem = FEM(P, T, N=order, BCs=BCs)
    fem.PlotDomain2D()

def check_hrefine():
     # Set polynomial order
     N = 1

     # Read and set up simple mesh
     filename = 'Test_Data/Maxwell05.neu'
     from ppp.MeshReader2D import MeshReaderGambitBC

     Nv, VX, VY, K, EToV, BCType = MeshReaderGambitBC(filename)
     P = np.array([VX, VY]).T
     T = EToV
     fem = FEM(P, T, N, BCType)
     # fem.plot_mesh()

     # make boundary conditions all Wall type
     fem.BCType = fem.BCs['Wall'] * (fem.EToE==(np.arange(K, dtype=int)[:, None] @ np.ones((1, fem.Nfaces))))

     # create a non-conforming interface by refinement
     refineflag = np.zeros((K,1))
     refineflag[:5] = 1
     fem.Hrefine2D(refineflag)
     fem.BuildHNonCon2D(N+1, 1e-6)

     # fem.plot_mesh()

     vec = np.arange(fem.Np * fem.K)[:, None]
     # print(vec)
     fem.NonConformingHFluxCorrection(vec=vec)

if __name__ == '__main__':
    # import dmsh, os
    # geo = dmsh.Rectangle(0, 1, 0, 1)
    # P, T = dmsh.generate(geo, .25) #h_max = .1
    # order = 3
    # fem = FEM(P, T, N=order)
    # fem.plot_mesh()
    # file_name = os.path.join('Test_Data', 'kovA02.neu')
    # multi_boundary(file_name)

    check_hrefine()




