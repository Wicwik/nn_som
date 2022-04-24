# This file was adopted from NN seminars and changed (to fit project purposes) by Robert Belanec
# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2022

import itertools

import numpy as np

from util import *



class SOM():

    def __init__(self, dim_in, n_rows, n_cols, inputs=None):
        self.dim_in = dim_in
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.history = None
        self.nonzero_idx = None
        self.classes = None
        self.class_counts = None

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
        '''
        return self.history:(size=(eps,4)) contains [alpha_t, lambda_t, Q_err, A]
        '''

        (_, count) = inputs.shape
        plot_in3d = self.dim_in > 2

        if live_plot:
            interactive_on()
            (plot_grid_3d if plot_in3d else plot_grid_2d)(inputs, self.weights, block=False)
            redraw()

        history = []
        for ep in range(eps):
            alpha_t  = alpha_s*(alpha_f/alpha_s)**(ep/(eps-1))
            lambda_t = lambda_s*(lambda_f/lambda_s)**(ep/(eps-1))

            qe_d = []
            w_prev = self.weights.copy()

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

                qe_d.append(np.linalg.norm(x - self.weights[win_r, win_c]))

            A = np.mean(np.linalg.norm(self.weights - w_prev, axis=2))

            Q_err = np.mean(qe_d)
            print('Ep {:3d}/{:3d}:  alpha_t = {:.3f}, lambda_t = {:.3f}, Q_err = {:.3f}, A = {:.3f}'
                  .format(ep+1, eps, alpha_t, lambda_t, Q_err, A))

            history.append([alpha_t/alpha_s, lambda_t/lambda_s, Q_err, A])

            if live_plot and ((ep+1) % live_plot_interval == 0):
                (plot_grid_3d if plot_in3d else plot_grid_2d)(inputs, self.weights, block=False)
                redraw()

        if live_plot:
            interactive_off()
        else:
            (plot_grid_3d if plot_in3d else plot_grid_2d)(inputs, self.weights, block=True)

        self.history = np.array(history)

    def evaluate(self, inputs, labels):
        '''
        Evaluate classes for each neuron
        Create array of neurons indexes that belong to some data
        Create list of classes for each neuron that belongs to som data (as its possible for neuron to have 2 classes the data structure is list of lists)
        Create list of counts for each neuron for each class
        This array and lists can help us predict test data and plot meshgrid
        returns (nonzero_idx, classes, counts)
        '''

        counts = np.zeros((self.n_rows, self.n_cols, int(np.max(labels))))

        (_, count) = inputs.shape
        for idx in range(count):
            x = inputs[:, idx]

            win_r, win_c = self.winner(x)
            counts[win_r, win_c, int(labels[idx])-1] += 1

        self.nonzero_idx = np.nonzero(np.sum(counts, axis=2))
        self.classes = []
        self.class_counts = []

        for c in counts[self.nonzero_idx]:
            idx = np.nonzero(c)[0]
            self.classes.append(list(idx))
            self.class_counts.append(list(c[idx].astype(int)))

        # print(list(zip(*self.nonzero_idx))[1], self.classes[1])

    def predict(self, inputs):
        predicted = []

        (_, count) = inputs.shape
        for idx in range(count):

            x = inputs[:, idx]

            # find winner neuron
            win_r, win_c = self.winner(x)
            # print((win_c, win_r))

            # get an array of neurons that belong to some class
            idx_arr = np.array(list(zip(*self.nonzero_idx)))
            # print(idx_arr)

            # find index of winner neuron in array of neurons that belong to some class
            pred_idx = np.where((idx_arr[:,0] == win_r) & (idx_arr[:,1] == win_c))

            # if the index was not found append None else append the class with the biggest count for that neuron
            if pred_idx[0].shape[0] == 0:
                predicted.append(-1)
            else:
                max_idx = np.argmax(self.class_counts[pred_idx[0][0]])
                predicted.append(self.classes[pred_idx[0][0]][max_idx])

        return np.array(predicted)

    def accuracy(self, predicted, targets):
        labels = targets-1

        CE = np.sum((predicted != labels))/labels.shape[0]
        return CE

    def get_u_matrix(self):
        '''
        Create U-Matrix in both directions
        We are also including neurons and distances between them
        '''

        # the U-Matrix will have size of (rows * 2 - 1, cols * 2 - 1) 
        self.u_matrix = np.zeros(shape=(self.n_rows * 2 - 1, self.n_cols * 2 - 1, 1), dtype=float)

        # first we calculate the L2 norm of neighboring neurons by iterating over all posible indexes
        for u_idx in itertools.product(range(self.n_rows * 2 - 1), range(self.n_cols * 2 - 1)):
            neigh = (0, 0)

            if not (u_idx[0] % 2) and (u_idx[1] % 2):
                neigh = (0, 1)  # horizontal distance
            
            if (u_idx[0] % 2) and not (u_idx[1] % 2):
                neigh = (1, 0)  # vertical distance

            # calcuate the distance between neigboring neurons, if both indexes are odd or even, we are calculating position of the neuron in the U-Matrix
            self.u_matrix[u_idx] = np.linalg.norm(self.weights[u_idx[0]//2, u_idx[1]//2] - self.weights[u_idx[0]//2 + neigh[0], u_idx[1]//2 + neigh[1]], axis=0)

        # we can calculate the positions of the neurons as the mean of neigbors
        for u_idx in itertools.product(range(self.n_rows * 2 - 1), range(self.n_cols * 2 - 1)):
            if not (u_idx[0] % 2) and not (u_idx[1] % 2):
                idx_list = []

                if u_idx[0] > 0:
                    idx_list.append((u_idx[0] - 1, u_idx[1]))
                
                if u_idx[0] < self.n_rows * 2 - 2:
                    idx_list.append((u_idx[0] + 1, u_idx[1])) 
                
                if u_idx[1] > 0:
                    idx_list.append((u_idx[0], u_idx[1] - 1)) 
                
                if u_idx[1] < self.n_cols * 2 - 2:
                    idx_list.append((u_idx[0], u_idx[1] + 1)) 

                self.u_matrix[u_idx] = np.mean([self.u_matrix[idx] for idx in idx_list])

            elif (u_idx[0] % 2) and (u_idx[1] % 2):
                idx_list = [(u_idx[0] - 1, u_idx[1]),(u_idx[0] + 1, u_idx[1]),(u_idx[0], u_idx[1] - 1),(u_idx[0], u_idx[1] + 1)]
                self.u_matrix[u_idx] = np.mean([self.u_matrix[idx] for idx in idx_list])

        return self.u_matrix