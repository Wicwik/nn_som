# Robert Belanec 4/2022
# 2nd NN project - Self Organizing Map (SOM)

from som import *
from util import *
import random

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
# inputs = np.random.rand(2, 250)
# inputs[1,:] += 0.5 * inputs[0,:]

# Iris input 
# inputs = np.loadtxt('iris.dat').T[:-1]
# labels = np.loadtxt('iris.dat').T[-1]

# Seeds input
inputs = np.loadtxt('seeds_dataset.txt').T[:-1]
labels = np.loadtxt('seeds_dataset.txt').T[-1]

(dim_in, count) = inputs.shape

# split train and test data
indices = np.arange(count)
random.shuffle(indices)
split = 150

print(inputs.shape)
train_indices = indices[:split] 
test_indices  = indices[split:]

train_inputs = inputs[:, train_indices]
train_labels = labels[train_indices]

test_inputs = inputs[:, test_indices]
test_labels = labels[test_indices]

## Train model

# Choose size of grid
# maximum should be rows*cols = size of train data
# CE around 10-15%
# rows = 5
# cols = 10

# rows = 10
# cols = 10

rows = 10
cols = 15

# number of neurons > than the number of training data (not recomended)
# rows = 20
# cols = 15

# Choose grid distance metric - L_1 / L_2 / L_max
grid_metric = L_1

# Some heuristics for choosing initial lambda
top_left = np.array((0, 0))
bottom_right = np.array((rows-1, cols-1))
lambda_s = grid_metric(top_left, bottom_right) * 0.5

plot_dots(inputs, labels, None, None, None, None, filename='all_data.png', title='All data')

# tran SOM on training data
model = SOM(dim_in, rows, cols, train_inputs)
model.train(train_inputs, eps=100, alpha_s=0.5, alpha_f=0.01, lambda_s=lambda_s, lambda_f=1, discrete_neighborhood=False, grid_metric=grid_metric)
model.evaluate(train_inputs, train_labels)

plot_history(model.history)
plot_mesh(rows, cols, model.nonzero_idx ,model.classes, model.class_counts, markersize=9)
plot_heatmaps(model.weights)
plot_u_matrix(np.squeeze(model.get_u_matrix()))

# get test prediction
preds = model.predict(test_inputs)

# compute accuracy
acc = model.accuracy(preds, test_labels)
print('Classification error: {:.3f}%'.format(acc*100))
