# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2022

import matplotlib
matplotlib.use('TkAgg') # fixme if plotting doesn`t work (try 'Qt5Agg' or 'Qt4Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm

import numpy as np
import atexit
import os
import time
import functools


## Utilities

def vector(array, row_vector=False):
    '''
    Construts a column vector (i.e. matrix of shape (n,1)) from given array/numpy.ndarray, or row
    vector (shape (1,n)) if row_vector = True.
    '''
    v = np.array(array)
    if np.squeeze(v).ndim > 1:
        raise ValueError('Cannot construct vector from array of shape {}!'.format(v.shape))
    return v.reshape((1, -1) if row_vector else (-1, 1))


def add_bias(X):
    '''
    Add bias term to vector, or to every (column) vector in a matrix.
    '''
    if X.ndim == 1:
        return np.concatenate((X, [1]))
    else:
        pad = np.ones((1, X.shape[1]))
        return np.concatenate((X, pad), axis=0)


def timeit(func):
    '''
    Profiling function to measure time it takes to finish function.
    Args:
        func(*function): Function to meassure
    Returns:
        (*function) New wrapped function with meassurment
    '''
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        start_time = time.time()
        out = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        ftime = elapsed_time
        msg = "Function [{}] finished in {:.3f} s"
        print(msg.format(func.__name__, ftime))
        return out
    return newfunc



## Interactive drawing

def clear():
    plt.clf()

def interactive_on():
    plt.ion()
    plt.show(block=False)
    time.sleep(0.1)

def interactive_off():
    plt.ioff()
    # plt.close()

def redraw():
    # plt.gcf().canvas.draw()   # fixme: uncomment if interactive drawing does not work
    plt.waitforbuttonpress(timeout=0.001)
    time.sleep(0.001)

def keypress(e):
    if e.key in {'q', 'escape'}:
        os._exit(0) # unclean exit, but exit() or sys.exit() won't work
    if e.key in {' ', 'enter'}:
        plt.close() # skip blocking figures

def use_keypress(fig=None):
    if fig is None:
        fig = plt.gcf()
    fig.canvas.mpl_connect('key_press_event', keypress)


## non-blocking figures still block at end

def finish():
    plt.show(block=True) # block until all figures are closed

atexit.register(finish)



## Plotting

palette = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']


def limits(values, gap=0.05):
    x0 = np.min(values)
    x1 = np.max(values)
    xg = (x1 - x0) * gap
    return np.array((x0-xg, x1+xg))


def plot_grid_2d(inputs, weights, i_x=0, i_y=1, s=60, block=True):
    fig = plt.figure(1)
    use_keypress(fig)
    fig.canvas.set_window_title('SOM neurons and inputs (2D)')
    plt.clf()

    plt.scatter(inputs[i_x,:], inputs[i_y,:], s=s, c=palette[-1], edgecolors=[0.4]*3, alpha=0.5)

    n_rows, n_cols, _ = weights.shape

    for r in range(n_rows):
        plt.plot(weights[r,:,i_x], weights[r,:,i_y], c=palette[0])

    for c in range(n_cols):
        plt.plot(weights[:,c,i_x], weights[:,c,i_y], c=palette[0])

    plt.xlim(limits(inputs[i_x,:]))
    plt.ylim(limits(inputs[i_y,:]))
    plt.tight_layout()
    plt.savefig('test2d.png', dpi=fig.dpi)
    plt.show(block=block)


def plot_grid_3d(inputs, weights, i_x=0, i_y=1, i_z=2, s=60, block=True):
    fig = plt.figure(2)
    use_keypress(fig)
    fig.canvas.set_window_title('SOM neurons and inputs (3D)')

    if plot_grid_3d.ax is None:
        plot_grid_3d.ax = Axes3D(fig)

    ax = plot_grid_3d.ax
    ax.cla()

    ax.scatter(inputs[i_x,:], inputs[i_y,:], inputs[i_z,:], s=s, c=palette[-1], edgecolors=[0.4]*3, alpha=0.5)

    n_rows, n_cols, _ = weights.shape

    for r in range(n_rows):
        ax.plot(weights[r,:,i_x], weights[r,:,i_y], weights[r,:,i_z], c=palette[0])

    for c in range(n_cols):
        ax.plot(weights[:,c,i_x], weights[:,c,i_y], weights[:,c,i_z], c=palette[0])

    ax.set_xlim(limits(inputs[i_x,:]))
    ax.set_ylim(limits(inputs[i_y,:]))
    ax.set_zlim(limits(inputs[i_z,:]))
    plt.savefig('test3d.png', dpi=fig.dpi)
    plt.show(block=block)

plot_grid_3d.ax  = None

