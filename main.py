import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn.metrics
import time
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from mlp_with_shortcut import Network, get_accuracy


def plot_history(history, legend, name):
    '''
    :param history: model train history dict with loss and test_accuracy
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
    plt.savefig(name)
    #plt.show()

if __name__ == '__main__':
    np.random.seed(42)
    data = np.array(pd.read_csv("data/car_data/car_evaluation_with_one_hot.csv"))
    data_x = data[:, 0:6]
    data_y = data[:, -4:]
    n_samples = data.shape[0]

    # TODO: find the best number of hidden layers
    dim_in = 6
    dim_hidden_1 = 6
    dim_hidden_2 = 14
    dim_out = 4

    learning_rate = 0.009
    n_epochs = 12
    batch_size = 15

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

    # parameters = {'dim_hidden_1':[4,6,8,10,12,20,50,100] ,
    #               'dim_hidden_2': [4, 6, 8,10,100,200] ,
    #               'learning_rate': [0.001, 0.005, 0.01],
    #               'n_epochs': [6,10,14,20],
    #               'batch_size': [6,12,20,32]}

    # model = Network(dim_in, dim_hidden_1, dim_hidden_2, dim_out, learning_rate, batch_size)
    # clf = GridSearchCV(model, parameters)
    # clf.fit(data_x_train, data_y_train)
    # res = sorted(clf.cv_results_.keys())
    # print(res)

    best_model_set = ""
    max_acc = 0
    acc_list = []
    model_desc_list = []
    for dim_hidden_1 in np.arange(2, 20, 2):
        for dim_hidden_2 in np.arange(6, 40, 4):
            for n_epochs in np.arange(9, 16, 3):
                for batch_size in np.arange(10, 21, 3):
                    history = []
                    legend = []
                    for learning_rate in np.arange(0.001, 0.01, 0.002):
                        legend.append("lr = {0:.3f}".format(learning_rate))
                        model = Network(dim_in, dim_hidden_1, dim_hidden_2, dim_out, learning_rate, batch_size)
                        history.append(model.fit(data_x_train, data_y_train, data_x_test, data_y_test, n_epochs, 'Softmax'))
                        y_test_pred = model.predict(data_x_test, 'Softmax')
                        test_acc = get_accuracy(data_y_test, y_test_pred)
                        model_desc = "{} lr, batch size {} and {} epochs. {} 1 hid, {} 2 hid".format(learning_rate, batch_size, n_epochs, dim_hidden_1, dim_hidden_2)
                        acc_list.append(test_acc)
                        model_desc_list.append(model_desc)
                        if test_acc > max_acc:
                            max_acc = test_acc
                            best_model_set = model_desc
                    plot_history(history, legend, "histories/net with batch size {} and {} epochs. {} 1 hid, {} 2 hid.png".format(batch_size, n_epochs, dim_hidden_1, dim_hidden_2))
    
    
    
    print(max_acc)
    print(best_model_set)
    print(acc_list)
    ind_max = np.asarray(acc_list).argsort()[-5:][::-1]
    print(ind_max)
    print(acc_list[ind_max])
    print(model_desc_list[ind_max])



    '''
    
    dim_hidden_1 = 6
    dim_hidden_2 = 14
    learning_rate = 0.009
    n_epochs = 12
    batch_size = 15
    
    model = Network(dim_in, dim_hidden_1, dim_hidden_2, dim_out, learning_rate, batch_size)
    start = time.time()
    history_softmax = model.fit(data_x_train, data_y_train, data_x_test, data_y_test, n_epochs, 'Softmax')
    end = time.time()
    print("Softmax activation lasted {} s".format(end - start))
    start = time.time()
    history_tanh = model.fit(data_x_train, data_y_train, data_x_test, data_y_test, n_epochs, 'Tanh')
    end = time.time()
    print("Tanh activation lasted {} s".format(end - start))
    #y_test_pred = model.predict(data_x_test)

    plt.figure(figsize=[12, 12])
    plt.subplot(211)
    plt.xlabel('epoch')
    plt.ylabel('Test accuracy')
    plt.title('Test Accuracy')
    plt.plot(history_softmax['test_accuracy'])
    plt.plot(history_tanh['test_accuracy'])
    plt.legend(['Softmax', 'Tanh'], loc='upper left')
    plt.savefig('Accuracy for Softmax vs Tanh activation with fixed seed.png')
    '''


'''
    y_train_pred = model.predict(data_x_train)
    train_acc = get_accuracy(data_y_train, y_train_pred)
    print("Train acc: {:3f}".format(train_acc))

    y_test_pred = model.predict(data_x_test)
    test_acc = get_accuracy(data_y_test, y_test_pred)
    print("Test acc: {:3f}".format(test_acc))
'''

'''

Train acc: 0.996691
Test acc: 0.955684

Best acc for model:
0.9633911368015414
0.009 lr, batch size 10 and 610 epochs dim1 =100, dim2 = ?200?



88
batch hid2 - 110, hid1 - 10? epoch - 16 batch - 10?

0.8766859344894027
0.009000000000000001 lr, batch size 15 and 12 epochs. 6 1 hid, 14 2 hid
'''
