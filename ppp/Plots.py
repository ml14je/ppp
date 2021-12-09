#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes : NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7 : Tue Nov  5 10:33:21 2019
"""
import numpy as np
import matplotlib.pyplot as pt

def plot_setup(x_label='', y_label='', x_log=False, y_log=False, bx=10, by=10,
               y_rev=False, scale=1, title='', my_dpi=100, project=None):
    """
    Sets up figure environment.

    Parameters
    ----------
    x_label : String
        Label of the x axis.
    y_label : String
        Label of the y axis.
    x_log : Boolean
        Determines whether the x axis is log scale.
    y_log : Boolean
        Determines whether the y axis is log scale.
    bx : Float
        The base of the x axis if x_log == True.
    by : Float
        The base of the y axis if y_log == True.
    scale : Float
        Scale of figure environment with respect to an aspect ratio of 13.68
        x 7.68.
    title : String
        Title of figure plot.
    dpi : Integer
        Density of pixels per square inch of figure plot.
    """
    import matplotlib.pyplot as pt
    import matplotlib as mpl

    mpl.style.use('seaborn-colorblind')

    fig, ax = pt.subplots(figsize=(13.68*scale, 7.68*scale),
                          dpi=my_dpi)
    set_axis(ax, x_label, y_label, x_log, y_log, bx, by,
               y_rev, scale, title)

    return fig, ax

def set_axis(ax, x_label='', y_label='', x_log=False, y_log=False, bx=10, by=10,
               y_rev=False, scale=1, title=''):
    import matplotlib.pyplot as pt
    pt.gca()
    ax.tick_params(axis="x", labelsize=16*scale)
    ax.tick_params(axis="y", labelsize=16*scale)

    ax.set_xlabel(x_label, fontsize=scale*20)
    ax.set_ylabel(y_label, fontsize=scale*20)
    ax.set_title(title, fontsize=scale*22)
    if x_log: ax.set_xscale('log', base=bx)
    if y_log: ax.set_yscale('log', base=by)
    # print(y_rev)
    # if y_rev: ax.invert_yaxis()
    ax.grid(linewidth=0.5)

def save_plot(fig, ax, file_name, folder_name=None, wd=None, my_dpi=100,
              my_loc='best', give_legend=True):
    import os

    if __name__ == '__main__':
        from File_Management import dir_assurer, file_exist
    else:
        from ppp.File_Management import dir_assurer, file_exist

    if not wd: wd = os.getcwd()

    def replace_all(text, dic):
        for i, j in dic.items():
            text = text.replace(i, j)
        return text

    def save(handles=None):
        if give_legend:
            try:
                handles, labels = ax.get_legend_handles_labels()

            except:
                AttributeError

            if handles: ax.legend(fontsize=16, loc=my_loc)

        fig.savefig(os.path.join(
                wd, os.path.join(
                        folder_name, file_name
                        )
                ),
                bbox_inches='tight', dpi=my_dpi)
        pt.close(fig)

    if folder_name:
        dir_assurer(folder_name, wd)
    else:
        folder_name = ''

    file_name = replace_all(
            file_name, {'\\' : '', '$' : '', ' ' : '_', ',' : '', '.' : ','}
            )

    file_name += '.png'
    if not file_exist(file_name, folder_name, wd):
        save()

    else:
        print('File already exists.')
        answer = input('\nWould you like to override old plot? Y for yes, and \
N for no: ').upper()

        if answer == 'Y':
            save()

        elif answer == 'N':
            return

        else:
            print('\nInvalid entry. Plot has not been saved.')
            return

def num_log(a):
    from math import log
    if a%1 ==0:
        return int(a)
    elif int(log(np.abs(a), 10)) == 0:
        return round(a, 2)
    else:
        return '{:.1E}'.format(a)

def num_log_str(a):
    from math import log
    if a%1 ==0:
        b =  str(int(a))
    elif int(log(np.abs(a), 10)) == 0:
        b = str(round(a, 2)).replace('.', ',')
    else:
        b = ('{:.1E}'.format(a)).replace('.', ',')

    return b

def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a

    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        num, den = (int(num/com), int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)

    return _multiple_formatter

@pt.FuncFormatter
def fake_log(x, pos):
    'The two args are the value and tick position'
    return f'$10^{{{x:.2f}}}$'

def subplots(rows, cols, scale=1, x_share=False, y_share=False, my_dpi=100):
    import matplotlib as mpl

    mpl.style.use('seaborn-colorblind')
    fig, axis = pt.subplots(rows, cols, figsize=(13.68*scale, 7.68*scale),
                          dpi=my_dpi, sharex=x_share, sharey=y_share)
    return fig, axis

if __name__=='__main__':
    x_arr = np.linspace(0, 2*np.pi, 100)
    y_arr = np.sin(x_arr)
    fig, ax = plot_setup('$x$', '$y$', scale=.5, title='$\sin{(x)}$')
    data = np.random.randint(10, size=(100,100))
    cp = ax.matshow(data, vmin=0, vmax=5)
    fig.colorbar(cp)

