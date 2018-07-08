import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn.metrics
import time
from mlp_with_shortcut import Network, get_accuracy


def plot_history_single(history, file_name):
    '''
    Plotting history of model fitting for a single parameter set.
    :param history: model train history dict with loss and test_accuracy
    :param file_name: name of file we save the figure in
    '''
    plt.figure(figsize=[12,12])
    plt.subplot(211)

    plt.plot(history['loss'])
    plt.xlabel('epoch')
    plt.ylabel('Log loss')
    plt.title('Log loss')

    plt.subplot(212)
    plt.plot(history['test_accuracy'])
    plt.xlabel('epoch')
    plt.ylabel('Test accuracy')
    plt.title('Test Accuracy')
    plt.savefig(file_name)
    #plt.show()


def plot_history_multiple(history, legend, file_name):
    '''
    Plotting history for multiple parameter sets.
    :param history: model train history dict with loss and test_accuracy
    :param legend: list of names of series passed for plotting
    :param file_name: name of file we save the figure in
    '''
    plt.figure(figsize=[12,12])
    plt.subplot(211)

    for hist in history:
        plt.plot(hist['loss'])
    plt.xlabel('epoch')
    plt.ylabel('Log loss')
    plt.title('Log loss')
    plt.legend(legend, loc='upper left')

    plt.subplot(212)
    for hist in history:
        plt.plot(hist['test_accuracy'])
    plt.xlabel('epoch')
    plt.ylabel('Test accuracy')
    plt.title('Test Accuracy')
    plt.legend(legend, loc='upper left')
    plt.savefig(file_name)
    #plt.show()


def custom_grid_search(dim_in, dim_out,
                       data_x_train, data_y_train, data_x_test, data_y_test,
                       log_grid, params, top_amount):

    best_model_set = ""
    max_acc = 0
    acc_list = []
    model_desc_list = []
    for dim_hidden_1 in params['dim_hidden_1']:
        for dim_hidden_2 in params['dim_hidden_2']:
            for n_epochs in params['n_epochs']:
                for batch_size in params['batch_size']:
                    history = []
                    legend = []
                    for learning_rate in params['learning_rate']:
                        legend.append("lr = {0:.3f}".format(learning_rate))
                        model = Network(dim_in, dim_hidden_1, dim_hidden_2, dim_out, learning_rate, batch_size)
                        history.append(
                            model.fit(data_x_train, data_y_train, data_x_test, data_y_test, n_epochs, 'Softmax'))
                        y_test_pred = model.predict(data_x_test, 'Softmax')
                        test_acc = get_accuracy(data_y_test, y_test_pred)
                        model_desc = "{0:.3f} lr, batch size {1} and {2} epochs. {3} 1 hid, {4} 2 hid".format(learning_rate,
                                                                                                     batch_size,
                                                                                                     n_epochs,
                                                                                                     dim_hidden_1,
                                                                                                     dim_hidden_2)
                        acc_list.append(test_acc)
                        model_desc_list.append(model_desc)
                        if test_acc > max_acc:
                            max_acc = test_acc
                            best_model_set = model_desc
                    plot_history_multiple(history, legend,
                                 "histories/net with batch size {} and {} epochs. {} 1 hid, {} 2 hid.png".format(
                                     batch_size, n_epochs, dim_hidden_1, dim_hidden_2))

    file = open(log_grid,"a")
    file.write("\n")
    file.write("------------- NEW RUN -----------------\n")
    ind_max = np.asarray(acc_list).argsort()[-top_amount:][::-1]
    for i in ind_max:
        file.write("Accuracy = {0:.3f} for model with this params:\n {1}\n".format(acc_list[i], model_desc_list[i]))
    file.write("\n")
    file.close()
    return [max_acc, best_model_set]


def compare_activations(dim_in, dim_hidden_1, dim_hidden_2, dim_out,
                        learning_rate, batch_size, n_epochs,
                        data_x_train, data_y_train, data_x_test, data_y_test,
                        output_file_path, plot_name):
    file = open(output_file_path,"w+")
    model = Network(dim_in, dim_hidden_1, dim_hidden_2, dim_out, learning_rate, batch_size)
    start = time.time()
    history_softmax = model.fit(data_x_train, data_y_train, data_x_test, data_y_test, n_epochs, 'Softmax')
    end = time.time()
    file.write("Softmax activation lasted {} s\n".format(end - start))
    start = time.time()
    history_tanh = model.fit(data_x_train, data_y_train, data_x_test, data_y_test, n_epochs, 'Tanh')
    end = time.time()
    file.write("Tanh activation lasted {} s\n".format(end - start))
    file.write("Plot may be found in file '{}'\n".format(plot_name))
    file.close()
    plt.figure(figsize=[12, 10])
    plt.xlabel('epoch')
    plt.ylabel('Test accuracy')
    plt.title('Test Accuracy')
    plt.plot(history_softmax['test_accuracy'])
    plt.plot(history_tanh['test_accuracy'])
    plt.legend(['Softmax', 'Tanh'], loc='upper left')
    plt.savefig(plot_name)


if __name__ == '__main__':
    np.random.seed(42)
    data = np.array(pd.read_csv("data/car_data/car_evaluation_with_one_hot.csv"))
    data_x = data[:, 0:6]
    data_y = data[:, -4:]
    n_samples = data.shape[0]

    dim_in = 6
    dim_hidden_1 = 50
    dim_hidden_2 = 100
    dim_out = 4

    learning_rate = 0.001
    n_epochs = 200
    batch_size = 20

    run_grid_search = False
    compare_activation = False

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

    # Building a model
    model = Network(dim_in, dim_hidden_1, dim_hidden_2, dim_out, learning_rate, batch_size)
    history = model.fit(data_x_train, data_y_train, data_x_test, data_y_test, n_epochs, 'Softmax')

    y_train_pred = model.predict(data_x_train, 'Softmax')
    train_acc = get_accuracy(data_y_train, y_train_pred)
    print("Train acc: {:3f}".format(train_acc))

    y_test_pred = model.predict(data_x_test, 'Softmax')
    test_acc = get_accuracy(data_y_test, y_test_pred)
    print("Test acc: {:3f}".format(test_acc))

    plot_history_single(history, 'plots/{0} epochs with batch size {1}.png'.format(n_epochs, batch_size))

    if run_grid_search:
        # Search best parameters
        params = {
                    'dim_hidden_1': np.arange(5, 100, 5),
                    'dim_hidden_2': np.arange(10, 200, 10),
                    'n_epochs': np.arange(7, 16, 3),
                    'batch_size': np.arange(10, 21, 3),
                    'learning_rate': np.arange(0.001, 0.01, 0.002)
                    }
        max_acc, best_model_desc = custom_grid_search(dim_in, dim_out,
                           data_x_train, data_y_train, data_x_test, data_y_test,
                           "log_files/log_grid_search.txt", params, 5)
    if compare_activation:
        # Test for different activation functions
        compare_activations(dim_in, dim_hidden_1, dim_hidden_2, dim_out,
                            learning_rate, batch_size, n_epochs,
                            data_x_train, data_y_train, data_x_test, data_y_test,
                            "log_files/activations_comparison.txt", 'Accuracy for Softmax vs Tanh activation with fixed seed.png')



# Some results (saved before log file appeared)
'''

Train acc: 0.996691
Test acc: 0.955684

Best acc for model:
0.9633911368015414
0.009 lr, batch size 10 and 610 epochs dim1 =100, dim2 = ?200?

0.88
batch hid2 - 110, hid1 - 10? epoch - 16 batch - 10?

0.8766859344894027
0.009000000000000001 lr, batch size 15 and 12 epochs. 6 1 hid, 14 2 hid
'''
