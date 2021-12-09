#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Thu Jun 10 17:21:38 2021
"""
import numpy as np

class symplectic(object):
    def __init__(self, f, g, u_0, v_0, t_0, t_N, N=100, method='Euler'):
        self.t_0, self.t_N, self.u_0, self.v_0 = t_0, t_N, np.copy(u_0), np.copy(v_0)
        self.N = N
        self.dt = (self.t_N - self.t_0)/self.N
        self.f, self.g = f, g
        method = method.upper()

        self.t_vals = np.linspace(self.t_0, self.t_N, self.N + 1)

        try:
            if len(u_0.shape) == 2 and u_0.shape[1] == 1:
                self.u_vals = np.zeros((self.N+1, u_0.shape[0], 1), dtype=complex)
                self.v_vals = np.zeros((self.N+1, v_0.shape[0], 1), dtype=complex)
            else:
                self.u_vals = np.zeros((self.N+1, len(u_0)), dtype=complex)
                self.v_vals = np.zeros((self.N+1, len(v_0)), dtype=complex)

        except AttributeError:
            self.u_vals, self.v_vals = np.zeros(self.N+1), np.zeros(self.N+1)
            self.u_vals[0], self.v_vals[0] = u_0, v_0

        if method == 'EULER':
            self.method = euler

        elif method == 'STROMER--VERLET':
            self.method = stromer_verlet

        elif method == 'RUTH3':
            self.method = ruth3

        else:
            raise ValueError("method argument is not defined.")

        self.integrate()

    def integrate(self):
        t, u, v, k = self.t_0, self.u_0, self.v_0, 0

        for k, t in enumerate(self.t_vals[:-1]):
            u, v = self.method(self.f, self.g, u, v, self.dt)
            self.u_vals[k+1], self.v_vals[k+1] = u, v

    def interpolate(self, t_vals,  method='linear'):
        from scipy.interpolate import interp1d
        val_max, val_min = np.max(t_vals), np.min(t_vals)
        assert val_max <= self.t_N
        assert val_min >= self.t_0

        self.y_func = interp1d(self.t_vals, self.y_vals, kind=method)
        self.y_vals = self.y_func(t_vals)

def euler(f, g, u, v, h):

    u1 = u + h * f(u, v)
    v1 = v + h * g(u1, v)

    return u1, v1

def stromer_verlet(f, g, u, v, h):
    from Newton_Raphson import newton_raphson

    implicit_u = lambda u1: u1 - (u + .5 * h * f(u1, v))
    u1 = newton_raphson(implicit_u, u)
    implicit_v = lambda v2 : v2 - (v + .5 * h * (g(u1, v) + g(u1, v2)))
    v2 = newton_raphson(implicit_v, v+1e-2)
    u2 = u1 + .5 * h * f(u1, v2)

    return u2, v2

def ruth3(f, g, u, v, h):
    coeffs = np.array([
        [1, 2/3, -2/3],
        [-1/24, 3/4, 7/24]
        ])

    for ai, bi in coeffs.T:
        u += h * bi * f(u, v)
        v += h * ai * g(u, v)

    return u, v

def test(f, g, exact, t0, tN, y0):
    from ppp.Plots import plot_setup
    import matplotlib.pyplot as pt
    from time import perf_counter
    u0, v0 = y0

    methods = ['Euler', 'Stromer--Verlet', 'Ruth3']
    N_vals = 10**np.linspace(0, 5, 11)
    average_errs = np.empty((len(methods), len(N_vals)))
    times = np.copy(average_errs)
    for i, method_input in enumerate(methods):
        print(method_input)
        for j, N_input in enumerate(N_vals.astype(int)):

            start = perf_counter()
            ode = symplectic(f, g, u_0, v_0, t_0, t_N, N=N_input, method=method_input)
            times[i, j] = perf_counter() - start
            u_exact, v_exact = exact(t_N)
            u_approx, v_approx = ode.u_vals[-1], ode.v_vals[-1]
            err = np.array([
                np.abs(u_exact - u_approx), np.abs(v_exact - v_approx)
                ])

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
    # t_0, t_N, y_0 = 2, 3, 1.0
    # function = lambda t, y : 1 + (t - y)**2
    # exact = lambda t : t + 1/(1-t)

    t_0, t_N = 0, 1
    f = lambda u, v : v
    g = lambda u, v: -u
    u_0, v_0 = 1.0, 0.0
    exact = lambda t : (np.cos(t), -np.sin(t))
    test(f, g, exact, t_0, t_N, (u_0, v_0))
    # ode = symplectic(f, g, u_0, v_0, t_0, t_N, N=100, method='Verlet')




    # time = np.linspace(t_0, t_N, 101)
    # import matplotlib.pyplot as pt
    # u_exact, v_exact = exact(ode.t_vals)
    # u_approx, v_approx = ode.u_vals, ode.v_vals
    # for exact, approx in zip([u_exact, v_exact], [u_approx, v_approx]):
    #     pt.plot(ode.t_vals, exact, 'r')
    #     pt.plot(ode.t_vals, approx, 'k:')
    #     pt.show()




    # Linear System of ODEs
    # t_0, t_N = 0, 2*np.pi
    # y_0 = np.array([[1.0], [0.0]])
    # function = lambda t, y : np.array([[0.0, -1.0], [1.0, 0.0]]) @ y
    # exact = lambda t : np.array([[np.cos(t)], [np.sin(t)]])

    # test(function, exact, t_0, t_N, y_0)