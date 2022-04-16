# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2022

import numpy as np

from util import *


class SOM():

    def __init__(self, dim_in, n_rows, n_cols, inputs=None):
        self.dim_in = dim_in
        self.n_rows = n_rows
        self.n_cols = n_cols

        self.weights = np.random.rand(self.n_rows, self.n_cols, self.dim_in)

        # create weights indexes for mesh distance calculation
        self.w_idx = np.stack(np.unravel_index(np.arange(self.n_rows * self.n_cols, dtype=int).reshape(self.n_rows, self.n_cols), 
            (self.n_rows, self.n_cols)), axis=2)

        if inputs is not None:
            ma = np.max(inputs, axis=1)
            mi = np.min(inputs, axis=1)

            self.weights = mi + self.weights * (ma - mi)


    def winner(self, x):
        '''
        Find winner neuron and return its coordinates in grid (i.e. its "index").
        Iterate over all neurons and find the neuron with the lowest distance to input x (np.linalg.norm).
        '''

        return np.unravel_index(np.argmin(np.linalg.norm(self.weights - x, axis=2)), (self.n_rows, self.n_cols))


    def train(self, 
              inputs,                                                  # Matrix of inputs - each column is one input vector
              eps=100,                                                 # Number of epochs
              alpha_s=0.01, alpha_f=0.001, lambda_s=None, lambda_f=1,  # Start & end values for alpha & lambda
              discrete_neighborhood=True,                              # Use discrete or continuous (gaussian) neighborhood function
              grid_metric=(lambda u,v:0),                              # Grid distance metric
              live_plot=False, live_plot_interval=10                   # Draw plots dring training process
             ):

        (_, count) = inputs.shape
        plot_in3d = self.dim_in > 2

        if live_plot:
            interactive_on()
            (plot_grid_3d if plot_in3d else plot_grid_2d)(inputs, self.weights, block=False)
            redraw()

        for ep in range(eps):
            alpha_t  = alpha_s*(alpha_f/alpha_s)**(ep/(eps-1))
            lambda_t = lambda_s*(lambda_f/lambda_s)**(ep/(eps-1))

            for idx in np.random.permutation(count):
                x = inputs[:, idx]

                win_r, win_c = self.winner(x)

                if discrete_neighborhood:
                    # this will return a map of distances after which we discreetly move only the < lambda_t
                    d = grid_metric(self.w_idx, np.array((win_r, win_c)), axis=(len(self.w_idx.shape)-1))
                    h_t = d < lambda_t
                    self.weights[h_t] += alpha_t*(x-self.weights[h_t])
                else:
                    d = grid_metric(self.w_idx, np.array((win_r, win_c)), axis=(len(self.w_idx.shape)-1))
                    h_t = np.exp(-np.power(d,2)/np.power(lambda_t,2)).reshape(self.n_rows, self.n_cols, 1)
                    self.weights += alpha_t*(x-self.weights)*h_t

            print('Ep {:3d}/{:3d}:  alpha_t = {:.3f}, lambda_t = {:.3f}'
                  .format(ep+1, eps, alpha_t, lambda_t))

            if live_plot and ((ep+1) % live_plot_interval == 0):
                (plot_grid_3d if plot_in3d else plot_grid_2d)(inputs, self.weights, block=False)
                redraw()

        if live_plot:
            interactive_off()
        else:
            (plot_grid_3d if plot_in3d else plot_grid_2d)(inputs, self.weights, block=True)