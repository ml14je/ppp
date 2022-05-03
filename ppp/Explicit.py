#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Tue Aug 25 15:48:15 2020
"""
import numpy as np

class explicit(object):
    def __init__(self, function, y_0, t_0, t_N, N=100, method='Heun',
                 nt=10, verbose=True):
        self.func = function
        self.t_0, self.t_N, self.y_0 = t_0, t_N, np.copy(y_0)
        self.N = N
        self.dt = (self.t_N - self.t_0)/self.N
        self.nt = N//nt
        self.nnt = nt

        self.t_vals = np.linspace(self.t_0, self.t_N, self.N + 1)

        try:
            if len(y_0.shape) == 2 and y_0.shape[1] == 1:
                self.y_vals = np.zeros((self.nt+1, y_0.shape[0], 1), dtype=complex)
                self.y_vals[0] = y_0
            else:
                self.y_vals = np.zeros((self.nt+1, len(y_0)), dtype=complex)
                self.y_vals[0] = y_0

        except AttributeError:
            self.y_vals = np.zeros(self.nt+1)
            self.y_vals[0] = y_0

        if method == 'Forward Euler':
            self.method = forward_euler

        elif method == 'Explicit Midpoint':
            self.method = explicit_midpoint

        elif method == 'Heun':
            self.method = heun

        elif method == 'Ralston':
            self.method = ralston

        elif method == 'RK3':
            self.method = rk3

        elif method == 'Heun3':
            self.method = heun3

        elif method == 'Ralston3':
            self.method = ralston3

        elif method == 'SSPRK3':
            self.method = ssprk3

        elif method == 'RK4':
            self.method = rk4

        elif method == '3/8 Rule':
            self.method = rule_3div8_4

        elif method == 'RK5':
            self.method = rk5

        else:
            raise ValueError("method argument is not defined.")

        self.integrate()

    def integrate(self):
        t, y, k = self.t_0, self.y_0, 0

        for k, t in enumerate(self.t_vals[:-1]):
            dy = self.method(self.func, t, y, self.dt)
            y += dy
            if (k +1) % 50 == 0:
                print(k, np.max(abs(y)), t)
            
            if (k+1) % self.nnt == 0:
                print(k, self.nnt)
                self.y_vals[k+1] = y

    def interpolate(self, t_vals,  method='linear'):
        from scipy.interpolate import interp1d
        val_max, val_min = np.max(t_vals), np.min(t_vals)
        assert val_max <= self.t_N
        assert val_min >= self.t_0

        self.y_func = interp1d(self.t_vals, self.y_vals, kind=method)
        self.y_vals = self.y_func(t_vals)

def forward_euler(f, t, y, h):
    k1 = f(t, y)

    return h*k1

def explicit_midpoint(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + .5 * h, y + h * .5 * k1)

    return h*k2

def heun(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h, y + h * k1)

    return h*(k1 + k2)/2

def ralston(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + (2/3) * h, y + h * (2/3) * k1)

    return h*(k1 + 3*k2)/4

def rk3(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + .5 * h, y + h * .5 * k1)
    k3 = f(t + h, y + h * (-k1 + 2 * k2))

    return h*(k1 + 4*k2 + k3)/6

def heun3(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + (1/3)* h, y + h * (1/3) * k1)
    k3 = f(t + (2/3) * h, y + h * (2/3) * k2)

    return h*(k1 + 3*k3)/4

def ralston3(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + .5 * h, y + h * .5 *k1)
    k3 = f(t + .75 * h, y + h * .75 *k2)

    return h*(2*k1 + 3*k2 + 4*k3)/9

def ssprk3(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h, y + h * k1)
    k3 = f(t + .5 * h, y + h * .25 *(k1 + k2))

    return h*(k1 + k2 + 4*k3)/6

def rk4(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + .5 * h, y + .5 * h * k1)
    k3 = f(t + .5 * h, y + .5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return h * (k1 +  2*k2 + 2*k3 + k4) / 6

def rule_3div8_4(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + (1/3) * h, y + h * (1/3) * k1)
    k3 = f(t + (2/3) * h, y + h * ((-1/3) * k1 + k2))
    k4 = f(t + h, y + h * (k1 - k2 + k3))

    return h*(k1 + 3*k2 + 3*k3 + k4)/8

def rk5(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + .25 * h, y + h * .25 * k1)
    k3 = f(t + .25 * h, y + h * (k1 + k2)/8)
    k4 = f(t + .5 * h, y + h * (-.5 * k2 + k3))
    k5 = f(t + .75 * h, y + h * (3 * k1 + 9 * k4)/16)
    k6 = f(t + h, y + h * (-3 * k1 + 2 * k2 + 12 * k3 - 12 * k4 + 8 * k5)/7)

    return h*(7*k1 + 32*k3 + 12*k4 + 32*k5 + 7*k6)/90

def test(func, exact, t0, tN, y0):
    from ppp.Plots import plot_setup
    import matplotlib.pyplot as pt
    from time import perf_counter

    methods = ['Forward Euler', 'Heun', 'Ralston', 'RK3', 'Heun3',
                 'Ralston3', 'SSPRK3', 'RK4', '3/8 Rule', 'RK5']
    N_vals = 10**np.linspace(0, 5, 11)
    average_errs = np.empty((len(methods), len(N_vals)))
    times = np.copy(average_errs)
    for i, method_input in enumerate(methods):
        print(method_input)
        for j, N_input in enumerate(N_vals.astype(int)):

            start = perf_counter()
            ode = explicit(function, y_0, t_0, t_N, N=N_input, method=method_input)
            times[i, j] = perf_counter() - start
            err = np.abs((exact(t_N) - ode.y_vals[-1]))

            if type(err) == float:
                average_errs[i, j] = err#np.abs((exact(t_N) - ode.y_vals[-1]))

            else:
                average_errs[i, j] = np.mean(err)

    fig, ax = plot_setup('$N$', 'Time [s]', x_log=True, y_log=True)
    lineObjects = ax.plot(N_vals, times.T, 'x-')
    ax.legend(iter(lineObjects), methods, fontsize=16)
    pt.show()

    fig, ax = plot_setup('$N$', 'Absolute Error', x_log=True, y_log=True)
    lineObjects = ax.plot(N_vals, average_errs.T, 'x-')
    ax.legend(iter(lineObjects), methods, fontsize=16)
    pt.show()

if __name__ == '__main__':
    # Scalar Example
    t_0, t_N, y_0 = 2, 3, 1.0
    function = lambda t, y : 1 + (t - y)**2
    exact = lambda t : t + 1/(1-t)
    test(function, exact, t_0, t_N, y_0)

    # Linear System of ODEs
    t_0, t_N = 0, 2*np.pi
    y_0 = np.array([[1.0], [0.0]])
    function = lambda t, y : np.array([[0.0, -1.0], [1.0, 0.0]]) @ y
    exact = lambda t : np.array([[np.cos(t)], [np.sin(t)]])
    test(function, exact, t_0, t_N, y_0)
