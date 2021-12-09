#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Thu Mar 11 21:03:34 2021
"""
import numpy as np

def fft2D():
    from scipy.fft import ifftn
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    N = 100
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex='col', sharey='row')
    xf = np.zeros((N,N))
    xf[0, 5] = 1
    xf[0, N-5] = 1
    Z = ifftn(xf)
    ax1.imshow(xf, cmap=cm.Reds)
    ax4.imshow(np.real(Z), cmap=cm.gray)
    xf = np.zeros((N, N))
    xf[5, 0] = 1
    xf[N-5, 0] = 1
    Z = ifftn(xf)
    ax2.imshow(xf, cmap=cm.Reds)
    ax5.imshow(np.real(Z), cmap=cm.gray)
    xf = np.zeros((N, N))
    xf[5, 10] = 1
    xf[N-5, N-10] = 1
    Z = ifftn(xf)
    ax3.imshow(xf, cmap=cm.Reds)
    ax6.imshow(np.real(Z), cmap=cm.gray)
    plt.show()

def fft2D2():
    Lx, Ly = 2, 1
    Nx, Ny = 2**10, 2**6
    x, y = np.linspace(0, Lx, Nx+1)[:-1], np.linspace(0, Ly, Ny+1)[:-1]
    # dx, dy = Lx/Nx, Ly/Ny

    F = np.ones((Nx, Ny), dtype=complex)
    for i in range(5):
        for j in range(5):
            Fx = np.exp(i*2j*np.pi*x/Lx)
            Fy = np.exp(j*2j*np.pi*y/Ly)

            import matplotlib.pyplot as pt
            pt.plot(x, Fx.real, 'r-')
            pt.plot(y, Fy.real, 'b-')
            pt.show()

            from scipy.fft import fft2
            F += Fx[:, None] * Fy[None, :]


            cp = pt.matshow(F.real, extent=[0, Lx, 0, Ly], origin='Lower', aspect='auto')
            pt.colorbar(cp)
            pt.show()

            A_n = fft2(F.imag)
            A_n *= 2/((Nx)*(Ny))
            A_n[np.abs(A_n)<1e-10] = 0
            # print(np.abs(A_n[:6, :6]))

            k1, k2 = np.where(np.abs(A_n[:6, :6])!=0)

            if len(k1) == len(k2) == 0:
                K = np.zeros((1, 2))
            else:
                print(k1, k2)
                print(k1.shape, k2.shape)
                K = np.concatenate((k1[:, None], k2[:, None]), axis=1)
                print(K)

            # for k in K:
                # print(k)
            # if K.size > 0:
            #     print(K)
                # print(K[0], K[1])

            # cp = pt.matshow(np.abs(A_n[:6, :6]),
            #                 extent=[0, 5, 0, 5], aspect=None,
            #                 origin='lower')
            # pt.colorbar(cp)
            # pt.show()

    # Fx = np.cos()

if __name__ == '__main__':
    # T = 4
    # f_sample = 5
    # t = np.linspace(0, T, 2**10)
    # L = 2*np.pi/T
    # f = 3*np.sin(3*L*t) - 6*np.sin(2*L*t) + np.cos(4*L*t)
    # y = (2*np.fft.rfft(f) / t.shape)[:5]
    # print(y.real, y.imag, np.abs(y))

    fft2D2()