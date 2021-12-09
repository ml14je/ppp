#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Tue May 25 17:10:30 2021
"""
import numpy as np
from scipy.sparse import csr_matrix as sp
from scipy.linalg import solve, inv

class FEM(object):
    def __init__(self, P, T, N=1):
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
        # print(self.x.shape, self.Fmask.shape, self.x[self.Fmask[:]].shape)
        self.Fx, self.Fy = self.x[self.Fmask.flatten()], self.y[self.Fmask.flatten()]

        # Create surface integral terms
        self.Lift2D()

        # calculate geometric factors
        self.GeometricFactors2D()
        self.Normals2D()

        # Build connectivity matrix
        self.Connect2D()

        self.BuildMaps2D()
        # self.BuildBCMaps2D()

        from pandas import read_csv
        self.Vr, self.Vs = GradVandermonde2D(self.N, self.r, self.s)
        self.Drw = solve((self.V @ self.V.T), (self.V @ self.Vr.T).T).T
        self.Dsw = solve((self.V @ self.V.T), (self.V @ self.Vs.T).T).T

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
        Fmask = self.Fmask.flatten()
        # Interpolate geometric factors to face nodes
        fxr, fxs = xr[Fmask, :], xs[Fmask, :]
        fyr, fys = yr[Fmask, :], ys[Fmask, :]
        # build normals
        # fid1 = (1:Nfp).T; fid2 = (Nfp+1:2*Nfp).T; fid3 = (2*Nfp+1:3*Nfp).T

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
                x1, y1 = self.x.T.flatten()[vidM], self.y.T.flatten()[vidM]
                x2, y2 = self.x.T.flatten()[vidP], self.y.T.flatten()[vidP]
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
        self.mapP, self.mapM = mapP.flatten('F'), mapM.flatten()
        self.mapB = np.where(self.vmapP==self.vmapM)[0]
        self.vmapB = self.vmapM[self.mapB]

    def BuildBCMaps2D(self):
        """
        Build specialized nodal maps for various types of
        boundary conditions, specified in BCType.
        """
        bct = self.BCType.T
        bnodes = np.ones((self.Nfp, 1)) @ bct.flatten('F')[:, None]
        bnodes = bnodes.flatten('F')
        In = 1; Out = 2; Wall = 3; Far = 4; Cyl = 5; Dirichlet = 6;
        Neumann = 7; Slip = 8;

        # find location of boundary nodes in face and volume node lists
        self.mapI = bnodes==In ; self.vmapI = self.vmapM[self.mapI]
        self.mapO = bnodes==Out ; self.vmapO = self.vmapM[self.mapO]
        self.mapW = bnodes==Wall ; self.vmapW = self.vmapM[self.mapW]
        self.mapF = bnodes==Far ; self.vmapF = self.vmapM[self.mapF]
        self.mapC = bnodes==Cyl ; self.vmapC = self.vmapM[self.mapF]
        self.mapD = bnodes==Dirichlet ; self.vmapD = self.vmapM[self.mapC]
        self.mapN = bnodes==Neumann ; self.vmapN = self.vmapM[self.mapN]
        self.mapS = bnodes==Slip ; self.vmapS = self.vmapM[self.mapS]

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
        oFx = self.Fx.reshape(self.Nfp, self.Nfaces*self.K)
        oFy = self.Fy.reshape(self.Nfp, self.Nfaces*self.K)

        from ppp.Plots import plot_setup
        import matplotlib.pyplot as pt

        fig, ax= plot_setup('$x$', '$y$', title='Mesh')
        ax.plot(oFx, oFy, 'k-')

        pt.show()

    def PlotField2D(self, Nout, xin, yin, uin, title='Field'):
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
        uout = interp @ uin

        # render and format solution field
        import plotly.figure_factory as ff
        # import plotly.io as pio
        # pio.renderers.default='svg'
        fig = ff.create_trisurf(x=xout.T.flatten(), y=yout.T.flatten(),
                                z=uout.T.flatten(), simplices=TRI,
                                title=title)
        fig.show()

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

        print()

        xout = interp @ xin
        yout = interp @ yin
        if len(uin.shape) == 2:
            uout = (interp @ uin.T).T

        elif len(uin.shape) == 3:
            uout = np.zeros((uin.shape[0], interp.shape[0], uin.shape[2]))
            for i, u in enumerate(uin):
                uout[i] = (interp @ u)

        elif len(uin.shape) == 4:
            uout = np.zeros((uin.shape[0], uin.shape[1], interp.shape[0], uin.shape[3]))
            # print(uin.shape)
            for j in range(uin.shape[0]):
                uvals = uin[j]
                for i, u in enumerate(uvals):
                    uout[j, i] = (interp @ u)

        else:
            raise ValueError('Invalid u_in shape size')

        return TRI, xout, yout, uout, interp

    def PlotContour2D(self, tri, x, y, u, levels):
        """
        Generic routine to plot contours for triangulated data
        """
        Nlevels = len(levels)
        v1, v2, v3 = tri.T
        u1, u2, u3 = u.T.flatten()[v1], u.T.flatten()[v2], u.T.flatten()[v3]
        x1, x2, x3 = x.T.flatten()[v1], x.T.flatten()[v2], x.T.flatten()[v3]
        y1, y2, y3 = y.T.flatten()[v1], y.T.flatten()[v2], y.T.flatten()[v3]

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

    def makeCylinder2D(self, faces, ra, xo, yo):
        """
        Purpose: Use Gordon-Hall blending with an isoparametric map to
        modify a list of faces so they conform to a cylinder
        of radius r centered at (xo,yo)
        """
        NCurveFaces = faces.shape[0]
        self.vflag = np.zeros(self.VX.shape)

        for n in range(NCurveFaces):
            #move vertices of faces to be curved onto circle
            k, f = faces[n, 0], faces[n, 1]
            v1 = self.EToV[k, f]
            v2 = self.EToV[k, f%self.Nfaces+1]
            theta1, theta2 = np.arctan(self.VY[v1], self.VX[v1]), np.arctan(self.VY[v2], self.VX[v2])
            newx1, newy1 = xo + ra*np.cos(theta1), yo + ra*np.sin(theta1)
            newx2, newy2 = xo + ra*np.cos(theta2), yo + ra*np.sin(theta2)

            # update mesh vertex locations
            self.VX[v1], self.VX[v2] = newx1, newx2
            self.VY[v1], self.VY[v2] = newy1, newy2

            # store modified vertex numbers
            self.vflag[v1], self.vflag[v2] = 1, 1

        # map modified vertex flag to each element
        self.vflag = self.vflag(self.EToV)

        # locate elements with at least one modified vertex
        ks = np.sum(self.vflag, axis=1) > 0
        # build coordinates of all the corrected nodes
        va, vb, vc = self.EToV[ks, 0].T, self.EToV[ks, 1].T, self.EToV[ks, 2].T
        self.x[:, ks] = .5*(-(self.r+self.s)*self.VX[va]+(1+self.r)*self.VX[vb]+(1+self.s)*self.VX[vc])
        self.y[:, ks] = .5*(-(self.r+self.s)*self.VY[va]+(1+self.r)*self.VY[vb]+(1+self.s)*self.VY[vc])

        for n in range(NCurveFaces): # deform specified faces
            k, f = faces[n, 0], faces[n, 1]
            if f==0:
                v1, v2 = self.EToV[k, 0], self.EToV[k, 1]
                vr = self.r

            elif f==1:
                v1, v2 = self.EToV[k, 1], self.EToV[k, 2]
                vr = self.s

            elif f==2:
                v1, v2 = self.EToV[k, 0], self.EToV[k, 2]
                vr = self.s

            fr = vr[self.Fmask[:, f]]
            x1, y1 = self.VX[v1], self.VY[v1]
            x2, y2 = self.VX[v2], self.VY[v2]

            # move vertices at end points of this face to the cylinder
            theta1 = np.arctan(y1-yo, x1-xo)
            theta2 = np.arctan(y2-yo, x2-xo)

            # check to make sure they are in the same quadrant

            if ((theta2 > 0) and (theta1 < 0)):
                theta1 += 2*np.pi

            if ((theta1 > 0) & (theta2 < 0)):
                theta2 += 2*np.pi

            # distribute N+1 nodes by arc-length along edge
            theta = .5*(theta1*(1-fr) + theta2*(1+fr))

            # evaluate deformation of coordinates
            fdx = xo + ra*np.cos(theta)-self.x[self.Fmask[:, f], k]
            fdy = yo + ra*np.sin(theta)-self.y[self.Fmask[:, f], k]

            # build 1D Vandermonde matrix for face nodes and volume nodes
            Vface, Vvol = Vandermonde1D(self.N, fr), Vandermonde1D(self.N, vr)

            # compute unblended volume deformations
            from scipy.sparse import spsolve
            vdx, vdy = Vvol @ spsolve(Vface, fdx), Vvol @ spsolve(Vface, fdy)

            # blend deformation and increment node coordinates
            ids = np.abs(1-vr) > 1e-7 # warp and blend
            if f==0:
                blend = -(self.r[ids]+self.s[ids])/(1-vr[ids])

            elif f==1:
                blend = +(self.r[ids]+1)/(1-vr[ids])

            elif f==2:
                blend = -(self.r[ids]+self.s[ids])/(1-vr[ids])

            self.x[ids, k] += blend * vdx[ids]
            self.y[ids, k] += blend * vdy[ids]

        # repair other coordinate dependent information
        self.Fx, self.Fy = self.x[self.Fmask.T.flatten(), :], self.y[self.Fmask.T.flatten(), :]
        self.GeometricFactors2D()
        self.Normals2D()
        self.Fscale()

    def BuildCurvedOPS2D(self, intN):
        from scipy.sparse import spsolve
        # 1. Create cubature information
        # 1.1 Extract cubature nodes and weights
        from Cubature import Cubature2D
        cR, cS, cW, Ncub = Cubature2D(intN)

        # 1.1. Build interpolation matrix (nodes->cubature nodes)
        cV = self.InterpMatrix2D(cR, cS)

        # 1.2 Evaluate derivatives of Lagrange interpolants at cubature nodes
        cDr, cDs = Dmatrices2D(self.N, cR, cS, self.V)

        # 2. Create surface quadrature information
        # 2.1 Compute Gauss nodes and weights for 1D integrals
        from Jacobi import JacobiGQ
        gz, gw = JacobiGQ(0, 0, intN)

        # 2.2 Build Gauss nodes running counter-clockwise on element faces
        gR = np.concatenate((gz, -gz, -np.ones(gz.shape)))
        gS = np.concatenate((-np.ones(gz.shape), gz, -gz))

        # 2.3 For each face
        for f1 in range(self.Nfaces):
            # 2.3.1 build nodes->Gauss quadrature interpolation and
            # differentiation matrices

            gV[:, :, f1] = self.InterpMatrix2D(gR[:, f1], gS[:, f1])
            gDr[:, :, f1], gDs[:, :, f1] = Dmatrices2D(self.N, gR[:, f1], gS[:, f1], self.V)

        # 3. For each curved element, evaluate custom operator matrices
        Ncurved = len(curved)

        # 3.1 Store custom information in array of Matlab structs
        cinfo = {}
        for c  in range(Ncurved):
            # find next curved element and the coordinates of its nodes
            k1 = curved[c]
            x1, y1 = self.x[:, k1], self.y[:, k1]
            cinfo[c] = {}
            cinfo[c]['elem'] = k1
            # compute geometric factors
            crx, csx, cry, csy, cJ = GeometricFactors2D(x1,y1,cDr,cDs);

            # build mass matrix
            cMM = cV.T @ np.diag(cJ * cW) @ cV
            cinfo[c]['MM'] = cMM

            # build physical derivative matrices
            cinfo[c]['Dx'] = spsolve(cMM, (cV.T @ np.diag(cW * cJ) @ (np.diag(crx) @ cDr + np.diag(csx) @ cDs)))
            cinfo[c]['Dy'] = spsolve(cMM, (cV.T @ np.diag(cW * cJ) @ (np.diag(cry) @ cDr + np.diag(csy) @ cDs)))

            # build individual lift matrices at each face
            for f1 in range(self.Nfaces):
                k2, f2 = self.EToE[k1, f1], self.EToF[k1, f1]

                # compute geometric factors
                grx,gsx,gry,gsy,gJ = GeometricFactors2D(x1, y1, gDr[:, :, f1], gDs[:, :, f1])

                # compute normals and surface Jacobian at Gauss points on face f1
                if f1==0:
                    gnx, gny= -gsx, -gsy

                elif f1==1:
                    gnx, gny = grx+gsx, gry+gsy

                elif f1==2:
                     gnx, gny = -grx, -gry

                gsJ = np.sqrt(gnx*gnx + gny*gny)
                gnx, gny = gnx/gsJ, gny/gsJ
                gsJ = gsJ*gJ

                # store normals and coordinates at Gauss nodes
                cinfo[c].gnx[:, f1], cinfo[c].gx[:, f1] = gnx,  gV[:, :, f1] * x1
                cinfo(c).gny[:, f1], cinfo[c].gy[:, f1] = gny, gV[:, :, f1] * y1

                # store Vandermondes for ’-’ and ’+’ traces
                cinfo(c).gVM[:, :, f1] = gV[:, :, f1]
                cinfo(c).gVP[:, :, f1] = gV[::-1, :, f2]

                # compute and store matrix to lift Gauss node data
                cinfo(c).glift[:, :, f1] = spsolve(cMM, (gV[:, :, f1].T @ np.diag(gw * gsJ)))

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
            # print(i, j, np.round(Simplex2DP(a, b, i, j), 4))
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

if __name__ == '__main__':
    pass