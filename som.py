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

        if inputs is not None:
            # "Fill" the input space with neurons - scale and shift neurons to inputs' distribution.
            # Note: SOM will train even without it, but it helps.
            ma = np.max(inputs, axis=1)
            mi = np.min(inputs, axis=1)

            self.weights = mi + self.weights * (ma - mi)


    def winner(self, x):
        '''
        Find winner neuron and return its coordinates in grid (i.e. its "index").
        Iterate over all neurons and find the neuron with the lowest distance to input x (np.linalg.norm).
        '''
        win_r, win_c = -1, -1

        for r in range(self.n_rows):
            for c in range(self.n_cols):
                norm = np.linalg.norm(self.weights[r, c] - x)
                win_norm = np.linalg.norm(self.weights[win_r, win_c] - x)
                if win_r == -1 and win_c == -1:
                    win_r = r
                    win_c = c
                    
                if norm < win_norm:
                    win_r = r
                    win_c = c


        return win_r, win_c


    def train(self,
              inputs,   # Matrix of inputs - each column is one input vector
              eps=100,  # Number of epochs
              alpha_s=0.01, alpha_f=0.001, lambda_s=None, lambda_f=1,  # Start & end values for alpha & lambda
              discrete_neighborhood=True,  # Use discrete or continuous (gaussian) neighborhood function
              grid_metric=(lambda u,v:0),  # Grid distance metric
              live_plot=False, live_plot_interval=10  # Draw plots dring training process
             ):

        (_, count) = inputs.shape
        plot_in3d = self.dim_in > 2

        if live_plot:
            interactive_on()
            (plot_grid_3d if plot_in3d else plot_grid_2d)(inputs, self.weights, block=False)
            redraw()

        for ep in range(eps):
            alpha_t  = alpha_s*(alpha_f/alpha_s)**(ep/(eps-1))  # FIXME
            lambda_t = lambda_s*(lambda_f/lambda_s)**(ep/(eps-1))  # FIXME

            for idx in np.random.permutation(count):
                x = inputs[:, idx]

                win_r, win_c = self.winner(x)

                # Use "d = grid_metric(vector_a, vector_b)" for grid distance
                # Use discrete neighborhood

                for r in range(self.n_rows):
                    for c in range(self.n_cols):
                        # ...
                        d = grid_metric(np.array((r, c)), np.array((win_r, win_c)))
                        h_t = 1 if d < lambda_t else 0

                        self.weights[r, c] += alpha_t*(x-self.weights[r, c])*h_t  # FIXME


            print('Ep {:3d}/{:3d}:  alpha_t = {:.3f}, lambda_t = {:.3f}'
                  .format(ep+1, eps, alpha_t, lambda_t))

            if live_plot and ((ep+1) % live_plot_interval == 0):
                (plot_grid_3d if plot_in3d else plot_grid_2d)(inputs, self.weights, block=False)
                redraw()

        if live_plot:
            interactive_off()
        else:
            (plot_grid_3d if plot_in3d else plot_grid_2d)(inputs, self.weights, block=True)