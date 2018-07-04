import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt

from mlp_with_shortcut import Network

if __name__ == '__main__':
    data = np.array(pd.read_csv("data/car_data/car_evaluation.csv"))
    data_x = data[:, 0:6]
    data_y = data[:, -1:]
    n_samples = data.shape[0]

    dim_in = 6
    dim_hidden_1 = 50
    dim_hidden_2 = 100
    dim_out = 4

    learning_rate = 1e-2
    n_epochs = 200
    batch_size = 100

    n_train = int(n_samples * 0.7)
    n_test = n_samples - n_train

    # train-test partition
    perm = np.random.permutation(n_samples)
    train_indx = perm[:n_train]
    test_indx = perm[n_train:]

    data_x_train, data_y_train = data_x[train_indx, :], data_y[train_indx]
    data_x_test, data_y_test = data_x[test_indx, :], data_y[test_indx]

    model = Network(dim_in, dim_hidden_1, dim_hidden_2, dim_out, learning_rate, batch_size)
    model.forward_pass(data_x_train[0])
    print(model.y_pred)