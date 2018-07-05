import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn.metrics

from mlp_with_shortcut import Network, get_accuracy


if __name__ == '__main__':
    data = np.array(pd.read_csv("data/car_data/car_evaluation_with_one_hot.csv"))
    data_x = data[:, 0:6]
    data_y = data[:, -4:]
    n_samples = data.shape[0]

    # TODO: find the best number of hidden layers
    dim_in = 6
    dim_hidden_1 = 100
    dim_hidden_2 = 50
    dim_out = 4

    learning_rate = 1e-4
    n_epochs = 300
    batch_size = 100

    n_train = int(n_samples * 0.7)
    n_test = n_samples - n_train

    # data shuffling
    perm = np.random.permutation(n_samples)

    # train-test partition
    train_indx = perm[:n_train]
    test_indx = perm[n_train:]

    data_x_train, data_y_train = data_x[train_indx, :], data_y[train_indx]
    data_x_test, data_y_test = data_x[test_indx, :], data_y[test_indx]

    data_x_train = data_x_train.T
    data_x_test = data_x_test.T

    model = Network(dim_in, dim_hidden_1, dim_hidden_2, dim_out, learning_rate, batch_size)
    history = model.fit(data_x_train, data_y_train, data_x_test, data_y_test, n_epochs)

    y_train_pred = model.predict(data_x_train)
    train_acc = get_accuracy(data_y_train, y_train_pred)
    print("Train acc: {:3f}".format(train_acc))

    y_test_pred = model.predict(data_x_test)
    test_acc = get_accuracy(data_y_test, y_test_pred)
    print("Test acc: {:3f}".format(test_acc))
