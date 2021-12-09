#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Wed Aug 26 10:02:29 2020
"""
import numpy as np

class implicit(object):
    def __init__(self, function, y_0, x_0, x_N, N=100,
                 method='Backward Euler', θ=None):
        self.func = function
        self.x_0, self.x_N, self.y_0 = x_0, x_N, y_0
        self.N = N
        self.dx = (self.x_N - self.x_0)/self.N

        self.x_vals = np.linspace(self.x_0, self.x_N, self.N + 1)

        try:
            if len(y_0.shape) == 2 and y_0.shape[1] == 1:
                self.y_vals = np.zeros((self.N+1, y_0.shape[0], 1), dtype=complex)
                self.scalar=False
            else:
                self.y_vals = np.zeros((self.N+1, y_0), dtype=complex)
                self.scalar=True

        except AttributeError:
            self.y_vals = np.zeros(self.N+1)
            self.y_vals[0] = y_0
            self.scalar=True

        if θ is not None and 0 <= θ <= 1:
            method = 'Theta Method'

        if method == 'Backward Euler':
            backward_euler = lambda f, t, y, h : theta_method(f, t, y, h, 0, scalar=self.scalar)
            self.method = backward_euler

        elif method == 'Crank-Nicolson':
            crank_nicolson = lambda f, t, y, h : theta_method(f, t, y, h, .5, scalar=self.scalar)
            self.method = crank_nicolson

        elif method == 'Theta Method':
            theta = lambda f, t, y, h : theta_method(f, t, y, h, θ, scalar=self.scalar)
            self.method = theta

        else:
            raise ValueError("method argument is not defined.")

        self.integrate()

    def integrate(self):
        x, y, h, k = self.x_0, self.y_0, self.dx, 0
        self.y_vals[0] = self.y_0

        for k, x in enumerate(self.x_vals):
            dy = self.method(self.func, x, y, h)
            y += dy
            self.y_vals[k] = y

    def interpolate(self, x_vals,  method='linear'):
        from scipy.interpolate import interp1d
        val_max, val_min = np.max(x_vals), np.min(x_vals)
        assert val_max <= self.x_N
        assert val_min >= self.x_0

        self.y_func = interp1d(self.x_vals, self.y_vals, kind=method)
        self.y_vals = self.y_func(x_vals)


def theta_method(f, t, y, h, θ, scalar=True):
    # θ = 1 corresponds to forward, θ=0 corresponds to backward while θ=.5
    # corresponds to Crank-Nicolson

    assert 0 <= θ <= 1

    if scalar:
        from ppp.Newton_Raphson import newton_raphson as newton

    else:
        from ppp.Newton_Raphson import newton_system as newton

    k1 = lambda y1 : f(t + h, y1)
    f1 = f(t, y)

    root = lambda y1 : y1 - y - h*(θ * f1 + (1 - θ) * k1(y1))

    y2 = newton(root, y, tolerance=1e-12)

    return h * (θ * f1 + (1 - θ) *k1(y2))


if __name__ == '__main__':
    # t_0, t_N, y_0 = 0, 10, 10.0
    # k = 2
    # function = lambda t, C : -k * C
    # exact = lambda t : y_0 * np.exp(-k * t)
    # ode = implicit(function, y_0, t_0, t_N, N=1000, method='Crank-Nicolson', θ=1)

    # time = np.linspace(t_0, t_N, 101)
    # import matplotlib.pyplot as pt
    # pt.plot(time, exact(time), 'r')
    # pt.plot(ode.x_vals, ode.y_vals, 'k:')
    # pt.show()

    # Linear System of ODEs
    t_0, t_N = 0, 2*np.pi
    time = np.linspace(t_0, t_N, 101)
    y_0 = np.array([[1.0], [0.0]])
    function = lambda t, y : (np.array([[0.0, -1.0], [1.0, 0.0]]) @ y[:, None]).T[0]
    exact = lambda t : np.array([[np.cos(t)], [np.sin(t)]])

    ode = implicit(function, y_0, t_0, t_N, N=1000, method='Crank-Nicolson', θ=0.5)

    import matplotlib.pyplot as pt
    pt.plot(time, exact(time), 'r')
    pt.plot(ode.x_vals, ode.y_vals, 'k:')
    pt.show()

