#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes : NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7 : Tue Nov  5 11:01:18 2019
"""

from Newton_Raphson import newton_raphson

def plus_func(u, h_val, x0, y0):
    """
    The plus stencil centered at (x0, y0) for uniform grid-spacing as
    determined by h_val.
    """
    return u(x0+h_val, y0)+u(x0, y0+h_val)+u(x0-h_val, y0)+u(x0, y0-h_val)

def cross_func(u, h_val, x0, y0):
    """
    The cross stencil centered at (x0, y0) for uniform grid-spacing as
    determined by h_val.
    """
    return u(x0+h_val, y0+h_val)+u(x0-h_val, y0+h_val)+\
                        u(x0-h_val, y0-h_val)+u(x0+h_val, y0-h_val)

def five_point_func(f, u, x0, y0, h_val):
    """
    Solves for u(x0, y0) in the the poisson equationΔu=f(x,y) and f specified
    using a five-point stencil with uniform-grid spacing as determined by
    h_val.
    
    Parameters
    ----------
    f : Function
        Function f as defined in Poisson's equation Δu=f(x,y).
    u : Function
        Function u as defined in Poisson's equation Δu=f(x,y).
    x0 : Float
        The x-value at which we attempt to evaluate u in Δu=f(x,y) for f(x, y)
        specified.
    y0 : Float
        The y-value at which we attempt to evaluate u in Δu=f(x,y) for f(x, y)
        specified.
    h_val : Float
        The uniform grid-spacing for which we wish to solve for u in
        Δu=f(x,y), for f(x, y) specified in problem.
    """
    
    u_val, f0 = u(x0, y0), f(x0, y0)
    plus= plus_func(u, h_val, x0, y0)
    
    def LHS(u0):
        return (plus-4*u0)/(h_val**2)
    
    RHS = f0
    
    g = lambda u0: LHS(u0)-RHS
    return newton_raphson(g, u_val)

def five_X_point_func(f, u, x0, y0, h_val):
    """
    Solves for u(x0, y0) in the the poisson equationΔu=f(x,y) and f specified
    using a five-point X stencil with uniform-grid spacing as determined by
    h_val.

    Parameters
    ----------
    f : Function
        Function f as defined in Poisson's equation Δu=f(x,y).
    u : Function
        Function u as defined in Poisson's equation Δu=f(x,y).
    x0 : Float
        The x-value at which we attempt to evaluate u in Δu=f(x,y) for f(x, y)
        specified.
    y0 : Float
        The y-value at which we attempt to evaluate u in Δu=f(x,y) for f(x, y)
        specified.
    h_val : Float
        The uniform grid-spacing for which we wish to solve for u in
        Δu=f(x,y), for f(x, y) specified in problem.
    """

    u_val, f0 = u(x0, y0), f(x0, y0)
    cross = cross_func(u, h_val, x0, y0)
    
    def LHS(u0):
        return (cross-4*u0)/(2*h_val**2)
    
    RHS = f0
    
    g = lambda u0: LHS(u0)-RHS
    return newton_raphson(g, u_val)

def nine_point_func(f, u, x0, y0, h_val):
    """
    Solves for u(x0, y0) in the the poisson equationΔu=f(x,y) and f specified
    using a nine-point stencil with uniform-grid spacing as determined by
    h_val.

    Parameters
    ----------
    f : Function
        Function f as defined in Poisson's equation Δu=f(x,y).
    u : Function
        Function u as defined in Poisson's equation Δu=f(x,y).
    x0 : Float
        The x-value at which we attempt to evaluate u in Δu=f(x,y) for f(x, y)
        specified.
    y0 : Float
        The y-value at which we attempt to evaluate u in Δu=f(x,y) for f(x, y)
        specified.
    h_val : Float
        The uniform grid-spacing for which we wish to solve for u in
        Δu=f(x,y), for f(x, y) specified in problem.
    """
    u_val, f0 = u(x0, y0), f(x0, y0)
    plus, cross = plus_func(u, h_val, x0, y0), cross_func(u, h_val, x0, y0)
    
    def LHS(u0):
        return (cross+4*plus-20*u0)/(6*h_val**2)
    
    RHS = f0
    g = lambda u0: LHS(u0)-RHS
    return newton_raphson(g, u_val)

def nine_point_mod_func(f, u, x0, y0, h_val):
    """
    Solves for u(x0, y0) in the the poisson equationΔu=f(x,y) and f specified
    using the nine-point stencil “Mehrstellenverfahren” with uniform-grid
    spacing as determined by h_val.

    Parameters
    ----------
    f : Function
        Function f as defined in Poisson's equation Δu=f(x,y).
    u : Function
        Function u as defined in Poisson's equation Δu=f(x,y).
    x0 : Float
        The x-value at which we attempt to evaluate u in Δu=f(x,y) for f(x, y)
        specified.
    y0 : Float
        The y-value at which we attempt to evaluate u in Δu=f(x,y) for f(x, y)
        specified.
    h_val : Float
        The uniform grid-spacing for which we wish to solve for u in
        Δu=f(x,y), for f(x, y) specified in problem.
    """
    u_val, f0 = u(x0, y0), f(x0, y0)
    plus, cross = plus_func(u, h_val, x0, y0), cross_func(u, h_val, x0, y0)
    
    def LHS(u0):
        return (cross+4*plus-20*u0)/(6*h_val**2)

    f_plus = f(x0+h_val, y0)+f(x0, y0+h_val)+f(x0-h_val, y0)+\
                        f(x0, y0-h_val)
    
    RHS = (f_plus+8*f0)/12
    g = lambda u0: LHS(u0)-RHS
    return newton_raphson(g, u_val)