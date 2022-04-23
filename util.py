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
palette_letters = ['b', 'r', 'g', 'y', 'c', 'm', 'k']


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

def plot_history(history, block=True):
    fig = plt.figure(3)
    use_keypress(fig)
    plt.clf()
    plt.ylim(bottom=0)
    fig.canvas.set_window_title('Metrics history of SOM training')

    ax = plt.subplot(1,2,1)
    ax.set_ylim(limits(history.T[:1]))
    plt.plot(history.T[0], label='alpha decay')
    plt.plot(history.T[1], label='lambda decay')
    plt.legend(loc='best')

    ax = plt.subplot(1,2,2)
    ax.set_ylim(limits(history.T[2:4]))
    plt.plot(history.T[2], label='avg. distance')
    plt.plot(history.T[3], label='avg. adjust')
    plt.legend(loc='best')


    plt.savefig('history.png', dpi=fig.dpi)
    plt.show(block=block)


def plot_mesh(rows, cols, nonzero_idx, classes, class_counts, block=True, markersize=10, figsize=(1918, 1025)):
    px = 1/plt.rcParams['figure.dpi']
    fig = plt.figure(4, figsize=tuple(i*px for i in figsize))
    use_keypress(fig)
    plt.clf()
    fig.canvas.set_window_title('Trained mesh')
    plt.title('Trained mesh')

    x, y = np.meshgrid(np.linspace(0, cols, cols), np.linspace(0, rows, rows))

    plt.plot(x, y, c='0.8', zorder=-1)
    plt.plot(np.transpose(x), np.transpose(y), c='0.8', zorder=-1)

    for i, n_i in enumerate(zip(*nonzero_idx)):
        # Single class markers
        if len(classes[i]) == 1:
            plt.plot(x[n_i], y[n_i], '{}o'.format(palette_letters[classes[i][0]]), markersize=markersize+class_counts[i][0]*3)

        # Two class markers
        elif len(classes[i]) == 2:
            plt.plot(x[n_i], y[n_i], '{}o'.format(palette_letters[classes[i][0]]), fillstyle='left', markersize=markersize+class_counts[i][0]*3)
            plt.plot(x[n_i], y[n_i], '{}o'.format(palette_letters[classes[i][1]]), fillstyle='right', markersize=markersize+class_counts[i][1]*3)

        # Three class markers
        elif len(classes[i]) == 3:
            plt.plot(x[n_i], y[n_i], '{}o'.format(palette_letters[classes[i][0]]), fillstyle='left', markersize=markersize+class_counts[i][0]*3)
            plt.plot(x[n_i], y[n_i], '{}o'.format(palette_letters[classes[i][1]]), fillstyle='right', markersize=markersize+class_counts[i][1]*3)
            plt.plot(x[n_i], y[n_i], '{}o'.format(palette_letters[classes[i][2]]), fillstyle='bottom', markersize=markersize+class_counts[i][2]*3)

        else:
            raise RuntimeError('Too many number of classes for one neuron. Use different size of mesh or optimize the training process when using plot_mesh function.')

    # plt.scatter(x[nonzero_idx], y[nonzero_idx], c=[item for sublist in classes for item in sublist])

    plt.axis('off')
    plt.savefig('mesh.png', dpi=fig.dpi)
    plt.show(block=block)


def plot_heatmaps(weights, cmap='RdYlBu', block=False):
    a = np.moveaxis(weights, -1, 0)

    for idx, comp in enumerate(a):
        fig = plt.figure(5+idx)
        use_keypress(fig)
        plt.clf()
        fig.canvas.set_window_title('Component-{0} heatmap'.format(idx))
        plt.title('Component-{0} heatmap'.format(idx))

        plt.imshow(comp, cmap=cmap)
        plt.axis('off')
        plt.savefig('component-{0}.png'.format(idx), dpi=fig.dpi)
        plt.show(block=block)

def plot_u_matrix(u_matrix, cmap='Greys', block=False, title='Umatrix', figsize=(1918, 1025)):
    px = 1/plt.rcParams['figure.dpi']
    fig = plt.figure(title or plt.gcf().number + 1, figsize=tuple(i*px for i in figsize))
    use_keypress(fig)
    plt.clf()
    fig.canvas.set_window_title('U-Matrix')
    plt.title('U-Matrix')

    plt.axis('off')
    plt.imshow(u_matrix, cmap=cmap)
    plt.savefig('umatrix.png', dpi=fig.dpi)
    plt.show()

def plot_dots(inputs, labels=None, predicted=None, test_inputs=None, test_labels=None, test_predicted=None, s=60, i_x=0, i_y=1, title=None, figsize=(1918, 1025), block=True, filename=None):
    px = 1/plt.rcParams['figure.dpi']
    fig = plt.figure(title or plt.gcf().number + 1, figsize=tuple(i*px for i in figsize))
    use_keypress()
    plt.clf()

    if inputs is not None:
        if labels is None:
            plt.gcf().canvas.set_window_title('Data distribution')
            plt.scatter(inputs[i_x,:], inputs[i_y,:], s=s, c=palette[-1], edgecolors=[0.4]*3, alpha=0.5, label='train data')

        elif predicted is None:
            plt.gcf().canvas.set_window_title('Class distribution')
            for i, c in enumerate(set(labels)):
                plt.scatter(inputs[i_x,labels==c], inputs[i_y,labels==c], s=s, c=palette[i], edgecolors=[0.4]*3, label='train cls {}'.format(c))

        else:
            plt.gcf().canvas.set_window_title('Predicted vs. actual')
            for i, c in enumerate(set(labels)):
                plt.scatter(inputs[i_x,labels==c], inputs[i_y,labels==c], s=2.0*s, c=palette[i], edgecolors=None, alpha=0.333, label='train cls {}'.format(c))

            for i, c in enumerate(set(labels)):
                plt.scatter(inputs[i_x,predicted==c], inputs[i_y,predicted==c], s=0.5*s, c=palette[i], edgecolors=None, label='predicted {}'.format(c))

        plt.xlim(limits(inputs[i_x,:]))
        plt.ylim(limits(inputs[i_y,:]))

    if test_inputs is not None:
        if test_labels is None:
            plt.scatter(test_inputs[i_x,:], test_inputs[i_y,:], marker='s', s=s, c=palette[-1], edgecolors=[0.4]*3, alpha=0.5, label='test data')

        elif test_predicted is None:
            for i, c in enumerate(set(test_labels)):
                plt.scatter(test_inputs[i_x,test_labels==c], test_inputs[i_y,test_labels==c], marker='s', s=s, c=palette[i], edgecolors=[0.4]*3, label='test cls {}'.format(c))

        else:
            for i, c in enumerate(set(test_labels)):
                plt.scatter(test_inputs[i_x,test_labels==c], test_inputs[i_y,test_labels==c], marker='s', s=2.0*s, c=palette[i], edgecolors=None, alpha=0.333, label='test cls {}'.format(c))

            for i, c in enumerate(set(test_labels)):
                plt.scatter(test_inputs[i_x,test_predicted==c], test_inputs[i_y,test_predicted==c], marker='s', s=0.5*s, c=palette[i], edgecolors=None, label='predicted {}'.format(c))

        if inputs is None:
            plt.xlim(limits(test_inputs[i_x,:]))
            plt.ylim(limits(test_inputs[i_y,:]))

    plt.legend()
    if title is not None:
        plt.gcf().canvas.set_window_title(title)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=fig.dpi)

    plt.show(block=block)
