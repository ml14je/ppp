#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Sat Aug 29 00:05:00 2020
"""
import numpy as np

class shooting_method(object):
    def __init__(self, x1, x2, vec0, der, bc, atol=1e-3, rtol=1e-3, N=100):
        self.x1, self.x2 = x1, x2
        self.λ0 = vec0[-1]
        self.vec0 = np.copy(vec0)
        self.der_func = der
        self.score = bc
        self.atol, self.rtol = atol, rtol
        self.h1 = (self.x2-self.x1)/N

    def bvp(self, x_vals=None, solutions=True):
        from ppp.Newton_Raphson import newton_raphson
        from ppp.Embedded import embedded

        def f(λ):
            v0 = np.copy(self.vec0)
            v0[-1] = λ

            solver = embedded(self.der_func, v0, self.x1, self.x2, rtol=self.rtol,
                        atol=self.atol, method='Cash–Karp')
            x = self.score(solver.y_vals[-1])

            return  x

        self.λ = newton_raphson(f, self.λ0, tolerance=1e-15)

        if solutions:
            self.vec0[-1] = self.λ
            # print(self.x1, self.x2)
            solver = embedded(self.der_func, self.vec0, self.x1, self.x2,
                rtol=1e-8, atol=1e-8, method='Cash–Karp')

            if x_vals is not None:
                self.solutions = solver.interpolate(x_vals)
                self.x_vals = x_vals

            else:
                self.solutions = solver.y_vals
                self.x_vals = solver.x_vals

        return self.λ

    def plot_solution(self, labels):
        from ppp.Plots import plot_setup
        import matplotlib.pyplot as pt

        fig, ax = plot_setup('$x$', '$\\mathbf{y}$')
        lineObjects = ax.plot(self.x_vals, self.solutions.real)
        ax.legend(iter(lineObjects), labels, fontsize=16)
        pt.show()

def test():
    from time import perf_counter
    x1, x2 = 0, 1

    def F(x, y):
        # y = [y, v, λ]
        yy = np.copy(y)
        yy[0] = y[1] #dy/dx = v
        yy[1] = -(y[2]**2) * y[0]  #dv/dx = -λ^2 y
        yy[2] = 0 # dλ/dx = 0

        return yy

    λ0 = np.pi*(1.25)
    bc = lambda yy : yy[0] #y(x2) = 0
    vec0 = np.array([0, 1, λ0])

    shooter = shooting_method(x1, x2, vec0, F, bc)
    start = perf_counter()
    root = shooter.bvp()
    print(f'Time: {perf_counter() - start:.3f} s')
    print(f'Eigenvalue: {root:.5f}')

    shooter.plot_solution(['$y$', '$dy/dx$', '$\\lambda$'])

def test3():
    M, A = 3, .5
    a = 5 #domain length
    λ0 = -.9 # λ = ω^2

    def F(x, y):
        dy = np.copy(y)

        #y = [v, w, λ]
        λ = y[2]
        dy[0] = y[1] #dv/dx = w
        dy[1] = -((1/A) * (1 + λ) * ((1 - M**2/λ)**(3/2)) - x**2)*y[0]  #dw/dx =-((1/A)*(1 + λ)*(1-M^2/λ)^(3/2) - y^2)v(y)
        dy[2] = 0 # dλ/dx = 0

        return dy

    bc = lambda y1 : y1[0]

    # vec1 = np.array([0, 1e-3, λ0], dtype=complex)
    # vec0 = np.array([λ0, vec1[1]])
    vec0 = np.array([0, 1e-3, λ0])
    shooter = shooting_method(-5, 5, vec0, F, bc, atol=1e-3, rtol=1e-3, N=100)

    from time import perf_counter
    start = perf_counter()
    root = shooter.bvp().real
    print(f'Time: {perf_counter() - start:.3f} s')
    print(f'Eigenvalue: {root:.10f}')

    shooter.plot_solution(['$v(y)$', '$dv/dx$', '$\\lambda=\\omega^2$'])
    print(shooter.solutions)

def test4():
    A = 5
    λ0 = -1 # λ = ω^2

    def F(x, y):
        dy = np.copy(y)

        #y = [v, w, λ]
        λ = y[2]
        dy[0] = y[1] #dv/dx = w
        dy[1] = -(4/A**2) * (λ - 4*x*(x - (1/np.cosh(x))**2))*y[0] #dw/dx =-((1/A)*(1 + λ)*(1-M^2/λ)^(3/2) - y^2)v(y)
        print(x, dy[1])
        dy[2] = 0 # dλ/dx = 0

        return dy

    bc = lambda y1 : y1[0]

    # vec1 = np.array([0, 1e-3, λ0], dtype=complex)
    # vec0 = np.array([λ0, vec1[1]])
    vec0 = np.array([0, 1e-3, λ0])
    shooter = shooting_method(-.5, 2, vec0, F, bc, atol=1e-3, rtol=1e-3, N=100)

    from time import perf_counter
    start = perf_counter()
    root = shooter.bvp().real
    print(f'Time: {perf_counter() - start:.3f} s')
    print(f'Eigenvalue: {root:.10f}')

    shooter.plot_solution(['$v(y)$', '$dv/dx$', '$\\lambda=\\omega^2$'])
    print(shooter.solutions)

class shooting_method2(object):
    def __init__(self, x1, x2, xf, vec1, vec2, vec0, der, bc, var, atol=1e-11,
                 rtol=1e-11, N=1000):
        self.x1, self.x2, self.xf = x1, x2, xf
        assert vec1[-1] == vec2[-1]
        self.var = var
        self.vec0 = vec0

        self.vec1, self.vec2 = np.copy(vec1), np.copy(vec2)
        self.der_func = der
        self.score = bc
        self.atol, self.rtol = atol, rtol
        self.h1, self.h2 = (self.xf-self.x1)/N, (self.x2-self.xf)/N

    def bvp(self, x_vals=None, solutions=True):
        from ppp.Newton_Raphson import newton_system
        from ppp.Embedded import embedded

        def f(X):
            v1, v2 = np.copy(self.vec1), np.copy(self.vec2)
            v1, v2 = self.var(X, np.copy(self.vec1), np.copy(self.vec2))


            solver1 = embedded(self.der_func, v1, self.x1, self.xf, rtol=self.rtol,
                        atol=self.atol, method='Cash–Karp')
            solver2 = embedded(self.der_func, v2, self.x2, self.xf, rtol=self.rtol,
                        atol=self.atol, method='Cash–Karp')

            y1, y2 = solver1.y_vals[-1], solver2.y_vals[-1]

            return  self.score(y1, y2)

        X, n = newton_system(f, self.vec0, tolerance=1e-14)
        self.λ = X[0]

        if solutions:
            self.vec1, self.vec2 = self.var(X, np.copy(self.vec1), np.copy(self.vec2))
            solver1 = embedded(self.der_func, self.vec1, self.x1, self.xf, rtol=self.rtol,
                        atol=self.atol, method='Cash–Karp')

            solver2 = embedded(self.der_func, self.vec2, self.x2, self.xf, rtol=self.rtol,
                        atol=self.atol, method='Cash–Karp')

            if x_vals is not None:
                sols1 = solver1.interpolate(x_vals[x_vals<=self.xf])
                sols2 = solver2.interpolate(x_vals[x_vals>self.xf])
                self.solutions = np.concatenate((sols1, sols2), axis=0)
                self.x_vals = x_vals

            else:
                self.solutions =  np.concatenate((solver1.y_vals, solver2.y_vals[-2::-1]), axis=0)
                self.x_vals = np.concatenate((solver1.x_vals, solver2.x_vals[-2::-1]), axis=0)

        return self.λ

    def plot_solution(self, labels):
        from ppp.Plots import plot_setup
        import matplotlib.pyplot as pt

        fig, ax = plot_setup('$x$', '$\\mathbf{y}$')
        ax.axvline(self.xf, color='k', linestyle=':')
        lineObjects = ax.plot(self.x_vals, self.solutions.real)
        ax.legend(iter(lineObjects), labels, fontsize=16)
        pt.show()

def test2():
    from time import perf_counter
    x1, x2, xf = 0, 1, .75

    def F(x, y):
        # y = [y, v, λ]
        yy = np.copy(y)
        yy[0] = y[1] #dy/dx = v
        yy[1] = -(y[2]**2) * y[0]  #dv/dx = -λ^2 y
        yy[2] = 0 # dλ/dx = 0

        return yy

    def vary(X, v1, v2):
        λ = X[0]
        v1[-1], v2[-1] = λ, λ
        v2[1] = X[1]

        return v1, v2

    λ0 = np.pi*(1.3)
    bc = lambda y1, y2 : (y1-y2)[:-1]

    vec1 = np.array([0, 1, λ0], dtype=complex)
    vec2 = np.array([0, -1, λ0], dtype=complex)
    vec0 = np.array([λ0, vec2[1]])
    shooter = shooting_method2(x1, x2, xf, vec1, vec2, vec0, F, bc, vary)
    start = perf_counter()
    root = shooter.bvp().real
    print(f'Time: {perf_counter() - start:.3f} s')
    print(f'Eigenvalue: {root:.10f}')

    shooter.plot_solution(['$y$', '$dy/dx$', '$\\lambda$'])

if __name__ == '__main__':
    test4()