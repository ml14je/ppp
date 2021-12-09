#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Tue Sep 14 15:36:22 2021
"""
import numpy as np

def quadtree_algorithm(P, T, seedings=None):
    if seedings is None: #refine all cells
        seedings = np.mean(P[T], axis=1)

    for k in range(seedings.shape[0]): #loop through each seed
        centroids = np.mean(P[T], axis=1) #mid-points in each triangular cell
        count = P.shape[0] #number of current nodes
        v = centroids-seedings[k] #vector going from seed k to each centroid
        D = np.diag(v @ v.T) #euclidean distance from seed k to each centroid
        ind = np.argmin(D) #identify the cell in which the seed is located
        x = P[T][ind, :, 0] # x coordinates of nodes making up located cell
        y = P[T][ind, :, 1] # y coordinates of nodes making up located cell

        #New coordinates to be appended to P
        x_new = .5 * np.array([x[0] + x[1], x[1] + x[2], x[2] + x[0]])
        y_new = .5 * np.array([y[0] + y[1], y[1] + y[2], y[2] + y[0]])
        P_new = np.array([x_new, y_new]).T

        T_refine = T[ind] #This cell triangle is identified
        T = np.delete(T, ind, axis=0) #and subsequently deleted
        indices = count + np.arange(3, dtype=int)
        T_new = np.array([
            [T_refine[0], indices[0], indices[2]],
            [indices[0], T_refine[1], indices[1]],
            [indices[2], indices[1], T_refine[2]],
            [indices[0], indices[1], indices[2]]
            ]) #New *four* cells which are made from existing cell
        P = np.concatenate((P, P_new), axis=0) #Update node vector
        T = np.concatenate((T, T_new), axis=0) #Update cell vector

    return P, T

def marching_front_algorithm(P, T, seedings=None):
    pass

def plot_grid(P, T, seedings=None):
    from ppp.Plots import plot_setup
    fig, ax = plot_setup('$x$', '$y$')

    cells = P[T]
    for cell in cells:
        ax.plot(cell[[0, 1], 0], cell[[0, 1], 1], 'k-')
        ax.plot(cell[[1, 2], 0], cell[[1, 2], 1], 'k-')
        ax.plot(cell[[2, 0], 0], cell[[2, 0], 1], 'k-')

    ax.plot(P[:, 0], P[:, 1], 'ro', markersize=3)

    if seedings is not None:
        ax.plot(seedings[:, 0], seedings[:, 1], 'bx', markersize=4)

    return fig, ax

if __name__ == '__main__':
    x0, xN, y0, yN = -.02, .02, 0, .04
    h_max = .2
    from ppp.File_Management import file_exist, dir_assurer
    dir_assurer('Meshes')
    mesh_name = f'{xN-x0}x{yN-y0}_hmax={h_max}'
    mesh_dir = f'Meshes/{mesh_name}'

    if not file_exist(mesh_dir+'.npz'):
        from ppp.Numpy_Data import save_arrays
        import dmsh
        geo = dmsh.Rectangle(x0, xN, y0, yN)
        P, T = dmsh.generate(geo, h_max)
        save_arrays(mesh_dir, [P, T])

    else:
        from ppp.Numpy_Data import load_arrays
        P, T = load_arrays(mesh_dir)

    # seedings = np.random.random(size=(100, 2))


    # P_refine, T_refine = quadtree_algorithm(P, T)
    # plot_grid(P_refine, T_refine)
    # P, T = P_refine, T_refine

    # x0, xN, y0, yN = -1, 1, 0, 1
    if mesh_name =='':
        mesh_name = f'Rectangle_{xN-x0}x{yN-y0}_h={h_max}_refinement=True_canyon=False'

    from ppp.File_Management import file_exist, dir_assurer
    dir_assurer('Meshes')
    mesh_dir = f'Meshes/{mesh_name}'
    w = 1e-2


    def h_func(x, y):
        hC = .025 # Coastal-shelf fluid depth
        H0, ΔH = (1+hC)/2, (1-hC)/2 # H0 is mean fluid depth on and off shelf, and ΔH is change in fluid depth over 2
        L0, ΔL = .02, .015 # L0 is shelf-width without canyon, and ΔL is canyon pertrusion
        α = 4 # Gradient change of Gaussian shelf-width profile
        λ = .5 # Slope-width parameter (Teeluck, 2013)
        xC = .5 * (x0 + xN) # Center of canyon in x
        # w = .02 # Width of canyon for α >> 1
        L = L0 - ΔL * np.exp(-(2*(x - xC)/w)**α/2) # Shelf-width variance in x
        return  H0 + ΔH * np.tanh((y-L)/(λ*L)) # Fluid depth - same as Teeluck, but varying L

    xvals, yvals= np.linspace(x0, xN, 1001), np.linspace(y0, yN, 1001)
    H = h_func(xvals[None, :], yvals[:, None])
    fig, ax = plot_grid(P, T)
    cm = ax.matshow(H, origin='lower', aspect='auto', extent=[x0, xN, y0, yN], alpha=.25,
                cmap='Blues')
    cb = fig.colorbar(cm)
    import matplotlib.pyplot as pt
    pt.show()

    tol = 1e-2
    while True:
        Xc, Yc = np.mean(P[T, 0], axis=1), np.mean(P[T, 1], axis=1)
        # print(np.mean(h_func(P[T, 0], P[T, 1]), axis=1))
        dx_vals = abs(h_func(Xc, Yc) - np.mean(h_func(P[T, 0], P[T, 1]), axis=1))
        dx = np.max(dx_vals)
        print()

        if dx < tol:
            break

        inds = dx_vals > tol
        # print(Xc, Yc, inds)
        seedings = np.array([Xc[inds], Yc[inds]]).T
        # print(dx)
        P_refine, T_refine = quadtree_algorithm(P, T, seedings)
        # pt.plot()
        fig, ax = plot_grid(P_refine, T_refine, seedings)
        cm = ax.matshow(H, origin='lower', aspect='auto', extent=[x0, xN, y0, yN], alpha=.25,
                    cmap='Blues')
        cb = fig.colorbar(cm)
        import matplotlib.pyplot as pt
        pt.show()
        P, T = P_refine, T_refine


        # print(inds, inds)
        # raise ValueError