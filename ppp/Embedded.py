#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Tue Aug 25 16:52:47 2020
"""
import numpy as np

class embedded(object):
    def __init__(self, function, y_0, t_0, t_N, rtol=1e-8, atol = 1e-8,
                 method='Heun', n_max=10000):
        self.func = function
        self.t_0, self.t_N, self.y_0 = t_0, t_N, y_0
        self.n = length(y_0)
        self.n_max = n_max #Max number of iterations
        self.safe = .95 #Safety factor
        self.EPS = 1e-16 #Machine Precision

        self.htry = (self.t_N-self.t_0)/100
        self.hmax, self.hmin = 10 * self.htry, .1 * self.htry
        self.rtol, self.atol = rtol, atol

        if method == 'Heun_Euler':
            self.method = heun_euler
            self.k = 2

        elif method == 'Runge–Kutta–Fehlberg':
            self.method = fehlberg
            self.k = 2

        elif method == 'Bogacki–Shampine':
            self.method = bogacki_shampine
            self.k = 3

        elif method == 'Fehlberg':
            self.method = fehlberg2
            self.k = 5

        elif method == 'Cash–Karp':
            self.method = cash_karp
            self.k = 5

        elif method == 'Dormand–Prince':
            self.method = dormand_prince
            self.k = 5

        else:
            raise ValueError("method argument is not defined.")

        self.β  = .4/self.k
        self.α = (1/self.k) - .75*self.β
        self.integrate()

    def integrate(self):
        t, self.y, k = self.t_0, self.y_0, 0
        t_vals, y_vals = [self.t_0], [self.y_0]
        self.reject = False
        self.err_old = self.rtol
        self.sgn = abs(self.htry)/self.htry

        while (k < self.n_max) or (self.sgn*t < self.sgn*self.t_N):
            self.h = self.htry
            while True:
                dy1, dy2 = self.method(self.func, t, self.y, self.h)
                self.y_err = dy1 - dy2
                self.y_out = self.y + dy1
                self.calc_error(dy1, dy2)
                k += 1

                if self.success():
                    break

                if abs(self.h) <= abs(t)*self.EPS:
                    raise ValueError('Step size underflow.')

            self.y = self.y_out
            t += self.h

            t_vals.append(t)
            y_vals.append(self.y)

            if self.sgn*t > self.sgn*self.t_N:
                #Interpolate
                from ppp.FD_Weights import weights
                w = weights(self.t_N, t_vals[-4:], 0)
                ys = np.array(y_vals[-4:])

                t_vals[-1], y_vals[-1] = self.t_N, (w.T @ ys)[0]
                
                break

        self.t_vals, self.y_vals = np.array(t_vals), np.array(y_vals)

    def interpolate(self, t_vals, method='linear'):
        from scipy.interpolate import interp1d
        val_max, val_min = np.max(t_vals), np.min(t_vals)
        assert self.sgn*val_max <= self.sgn*self.t_N
        assert self.sgn*val_min >= self.sgn*self.t_0

        y_interp = np.zeros((len(t_vals), self.n), dtype=complex)
        for i in range(self.n):
            func = interp1d(self.t_vals, self.y_vals[:, i], kind=method)
            y_interp[:, i] = func(t_vals)

        return y_interp

    def calc_error(self, dy1, dy2):
        from math import sqrt

        if self.n > 1:
            err = 0
            for i in range(self.n):
                sk = self.atol + self.rtol * max(abs(self.y[i]), abs(self.y_out[i]))
                err += ((self.y_err[i]/sk)*(self.y_err[i]/sk).conjugate()).real

            self.err = sqrt(err/self.n)

        else:
            sk = self.atol + self.rtol * max(abs(self.y), abs(self.y_out))
            self.err = np.abs(self.y_err/sk)

    def success(self):
        if self.err <= 1:
            if self.err == 0:
                self.scale = self.hmax

            else:
                self.scale = self.safe * self.h * (self.err**-self.α) * \
                                (self.err_old**self.β)

                if self.scale < self.hmin:
                    self.scale = self.hmin

                if self.scale > self.hmax:
                    self.scale = self.hmax

            if self.reject:
                self.hnext = self.h * min(self.scale, 1)

            else:
                self.hnext = self.h * self.scale

            self.err_old = max(self.err, self.rtol)
            self.reject = False
            return True

        else:
            self.scale = max(self.safe*(self.err**-self.α), self.hmin)
            self.reject = True
            self.h *= self.scale
            return False

def length(l):
    try:
        return len(l)

    except TypeError:
        return 1

def heun_euler(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h, y + h * k1)

    dy1 = h*(k1 + k2)/2
    dy2 = h*k1

    return dy1, dy2

def fehlberg(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + .5 * h, y + h * .5 * k1)
    k3 = f(t + h, y + h * (k1 + 255 * k2)/256)

    dy1 = h*(k1 + 510*k2 + k3)/512
    dy2 = h*(k1 + 255*k2)/256

    return dy1, dy2

def bogacki_shampine(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + .5 * h, y + h * .5 * k1)
    k3 = f(t + .75 * h, y + h * .75 * k2)
    k4 = f(t + h, y + h * (2*k1 + 3*k2 + 4*k3)/9)

    dy1 = h*(2*k1 + 3*k2 + 4*k3)/9
    dy2 = h*(7*k1 + 6*k2 + 8*k3 + 3*k4)/24

    return dy1, dy2

def fehlberg2(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + .75 * h, y + h * .75 * k1)
    k3 = f(t + (3/8) * h, y + h * (3*k1 + 9*k2)/32)
    k4 = f(t + (12/13) * h, y + h * (1932*k1 - 7200*k2 + 7296*k3)/2197)
    k5 = f(t + h, y + h * ((439/216)*k1 - 8*k2 + (3680/513)*k3 - (845/4104)*k4))
    k6 = f(t + .5 * h, y + h * ((-8/27)*k1 + 2*k2 - (3544/2565)*k3 + (1859/4104)*k4 - (11/40)*k5))

    dy1 = h*((16/135)*k1 +	(6656/12825)*k3 + (28561/56430)*k4 - (9/50)*k5 + (2/55)*k6)
    dy2 = h*((25/216)*k1 + (1408/2565)*k3 + (2197/4104)*k4 - (1/5)*k5)

    return dy1, dy2

def cash_karp(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + (1/5) * h, y + h * (1/5) * k1)
    k3 = f(t + (3/10) * h, y + h * (3*k1 + 9*k2)/40)
    k4 = f(t + (3/5) * h, y + h * (3*k1 - 9*k2 + 12*k3)/10)
    k5 = f(t + h, y + h * ((-11/54)*k1 + (5/2)*k2 - (70/27)*k3 + (35/27)*k4))
    k6 = f(t + (7/8) * h, y + h * ((1631/55296)*k1 + (175/512)*k2 + (575/13824)*k3 + (44275/110592)*k4 + (253/4096)*k5))

    dy1 = h*((37/378)*k1 + (250/621)*k3 + (125/594)*k4 + (512/1771)*k6)
    dy2 = h*((2825/27648)*k1 + (18575/48384)*k3 + (13525/55296)*k4 + (277/14336)*k5 + .25*k6)

    return dy1, dy2

def dormand_prince(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + .2 * h, y + h * .2 * k1)
    k3 = f(t + .3 * h, y + h * (3*k1 + 9*k2)/40)
    k4 = f(t + .8 * h, y + h * (44*k1 - 168*k2 + 160*k3)/45)
    k5 = f(t + (8/9) * h, y + h * ((19372/6561)*k1 - (25360/2187)*k2	+ (64448/6561)*k3 - (212/729)*k4))
    k6 = f(t + h, y + h * ((9017/3168)*k1 - (355/33)*k2 + (46732/5247	)*k3 + (49/176)*k4 - (5103/18656)*k5))
    k7 = f(t + h, y + h * ((35/384)*k1 + (500/1113)*k3 + (125/192)*k4  - (2187/6784)*k5 + (11/84)*k6))

    dy1 = h*((35/384)*k1 + (500/1113)*k3 + (125/192)*k4 - (2187/6784)*k5 + (11/84)*k6)
    dy2 = h*((5179/57600)*k1 + (7571/16695)*k3 + (393/640)*k4 - (92097/339200)*k5 + (187/2100)*k6 + (1/40)*k7)

    return dy1, dy2

def test(func, exact, t_0, t_N, y_0):
    from ppp.Plots import plot_setup
    import matplotlib.pyplot as pt
    from time import perf_counter

    methods = ['Heun', 'Runge–Kutta–Fehlberg', 'Bogacki–Shampine', 'Fehlberg',
               'Cash–Karp', 'Dormand–Prince']
    tolerances = 10**np.linspace(-1, -15, 15)
    times = np.zeros((len(methods), len(tolerances)))
    for i, method_input in enumerate(methods):
        for j, tol in enumerate(tolerances):
            start = perf_counter()
            embedded(function, y_0, t_0, t_N, method=method_input,
                           rtol=tol, atol=tol)
            times[i, j] = perf_counter() - start

    fig, ax = plot_setup('Tolerance', 'Time [s]', x_log=True)
    lineObjects = ax.plot(tolerances, times.T, 'x-')
    ax.legend(iter(lineObjects), methods, fontsize=16)

    pt.show()

if __name__ == '__main__':
    t_0, t_N, y_0 = 0, 5, np.array([1, 0])

    def function(t, x):
        xx = np.copy(x)
        xx[0] = x[1]
        xx[1] = -x[0]*(2*np.pi/(t_N-t_0))**2

        return xx
    # exact = lambda t : (y_0 + .5) * np.exp(-t) + .5 * (np.sin(t) - np.cos(t))

    # test(function, exact, t_0, t_N, y_0)
    from ppp.Plots import plot_setup
    import matplotlib.pyplot as pt
    fig, ax = plot_setup('$t$', '$y(t)$')
    sols = embedded(function, y_0, t_0, t_N, method='Cash–Karp',
                           rtol=1e-10, atol=1e-10)
    ax.plot(sols.x_vals, sols.y_vals)
    pt.show()

