from som import *
from util import *

def L_max(i, j, axis=0):
    return np.max(np.abs(i - j), axis=axis)

def L_1(i, j, axis=0):
    # print(j,axis)
    return np.sum(abs(i - j), axis=axis)

def L_2(i, j, axis=0):
    return np.sqrt(np.sum(np.power(i - j, 2), axis=axis))

# Test norms
# A = np.array([1,2,3,10])
# B = np.array([2,3,4,5])
# print(np.linalg.norm(A - B), L_2(A,B))
# print(np.linalg.norm(A - B, ord=1), L_1(A,B))

# Test input
inputs = np.random.rand(2, 250)
inputs[1,:] += 0.5 * inputs[0,:]

# inputs = np.loadtxt('seeds_dataset.txt').T[:-1]

(dim_in, count) = inputs.shape

## Train model

# Choose size of grid
rows = 13
cols = 17

# rows = 23
# cols = 27

# Choose grid distance metric - L_1 / L_2 / L_max
grid_metric = L_1

# Some heuristics for choosing initial lambda
top_left = np.array((0, 0))
bottom_right = np.array((rows-1, cols-1))
lambda_s = grid_metric(top_left, bottom_right) * 0.5

model = SOM(dim_in, rows, cols, inputs)
model.train(inputs, eps=100, alpha_s=0.5, alpha_f=0.01, lambda_s=lambda_s, lambda_f=1, discrete_neighborhood=False, grid_metric=grid_metric)