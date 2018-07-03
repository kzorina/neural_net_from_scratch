import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt

from mlp_with_shortcut import Network

if __name__ == '__main__':
    data = pd.read_csv("data/car_data/car.data", sep=',', header=None)
    le = preprocessing.LabelEncoder()
    data = data.apply(le.fit_transform)
    data = np.asarray(data)
    data_x = data[:, 0:6]
    data_y = data[:, 6]
    n_samples = data.shape[0]
    dim_in = 6
    dim_out = 4

    n_train = int(n_samples * 0.7)
    n_test = n_samples - n_train

    # train-test partition
    perm = np.random.permutation(n_samples)
    train_indx = perm[:n_train]
    test_indx = perm[n_train:]

    data_x_train, dataY_train = data_x[train_indx, :], data_y[train_indx]
    data_y_test, dataY_test = data_x[test_indx, :], data_y[test_indx]

    dim_hidden1 = 100
    dim_hidden2 = 150
    learning_rate = 1e-2
    n_iteration = 1000

   # model = Network(dim_in, dim_hidden1, dim_hidden2, dim_out)