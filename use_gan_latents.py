from som import *
from util import *
import random
import h5py
import pickle

def L_max(i, j, axis=0):
    return np.max(np.abs(i - j), axis=axis)

def L_1(i, j, axis=0):
    # print(j,axis)
    return np.sum(abs(i - j), axis=axis)

def L_2(i, j, axis=0):
    return np.sqrt(np.sum(np.power(i - j, 2), axis=axis))

inputs_path = '../data/generated_images/latents/sample_z.h5'
labels_path = '../data/predictions/predictions_resnet34_eyeglasses.pkl'

inputs = None
with h5py.File(inputs_path, 'r') as f:
    inputs = f['z'][:].T

labels = None
with open(labels_path,'rb') as f:
        labels = np.round(pickle.load(f))

(dim_in, count) = inputs.shape

indices = np.arange(count)
random.shuffle(indices)
split = 1000

print(inputs.shape)
train_indices = indices[:split] 
test_indices  = indices[split:]

train_inputs = inputs[:, train_indices]
train_labels = labels[train_indices]

test_inputs = inputs[:, test_indices]
test_labels = labels[test_indices]

rows = 20
cols = 50

grid_metric = L_1

top_left = np.array((0, 0))
bottom_right = np.array((rows-1, cols-1))
lambda_s = grid_metric(top_left, bottom_right) * 0.5

# plot_dots(train_inputs, train_labels, None, test_inputs, test_labels, None, filename='all_data.png', title='All data')

# tran SOM on training data
model = SOM(dim_in, rows, cols, train_inputs)
model.train(train_inputs, eps=100, alpha_s=0.5, alpha_f=0.01, lambda_s=lambda_s, lambda_f=1, discrete_neighborhood=False, grid_metric=grid_metric)
model.evaluate(train_inputs, train_labels)

print(train_labels)
plot_history(model.history)
plot_mesh(rows, cols, model.nonzero_idx ,model.classes, model.class_counts, markersize=9)
# plot_heatmaps(model.weights)
plot_u_matrix(np.squeeze(model.get_u_matrix()))

# get test prediction
# preds = model.predict(test_inputs)

# compute accuracy
# acc = model.accuracy(preds, test_labels)
# print('Classification error: {:.3f}%'.format(acc*100))