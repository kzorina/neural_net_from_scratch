import numpy as np
from activation_functions import Tanh, Relu, Softmax
# TODO: test whether our activations work ok - compare with sklearn
from sklearn.neural_network._base import softmax, relu, tanh
# TODO: check whether out cross_entropy loss works correctly
from sklearn.metrics import log_loss
import tqdm
from sklearn.preprocessing import OneHotEncoder

class Network:
    """
    Constructs a feed-forward multilayer perceptron with two hidden layers and
    ResNet-like shortcut connections.

    Architecture:
    output = softmax((ReLU(tanh(w1*input + b1)*w2 + b2) + ws*x)*w_out + b3)
    """

    def __init__(self, dim_input, dim_hidden_1, dim_hidden_2, dim_output, learning_rate, batch_size=1):
        self.dim_input = dim_input
        self.dim_hidden_1 = dim_hidden_1
        self.dim_hidden_2 = dim_hidden_2
        self.dim_output = dim_output
        self.xavier_init()

        # TODO: find out where to use batch size
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    # TODO: test whether we really have Var == 1 for networks
    def xavier_init(self):
        # followed the next explanation of Xavier initialization
        # https://www.quora.com/What-is-an-intuitive-explanation-of-the-Xavier-Initialization-for-Deep-Neural-Networks

        # specifying estimated Variance of networks on each layer according to  'rule of thumb'
        var_net_1 = 2. / (self.dim_input + self.dim_hidden_1)
        var_net_2 = 2. / (self.dim_hidden_1 + self.dim_hidden_2)
        var_net_out = 2. / (self.dim_hidden_2 + self.dim_output)

        self.w1 = np.random.randn(self.dim_hidden_1, self.dim_input) * np.sqrt(var_net_1)
        self.w2 = np.random.randn(self.dim_hidden_2, self.dim_hidden_1) * np.sqrt(var_net_2)
        self.w_out = np.random.randn(self.dim_output, self.dim_hidden_2) * np.sqrt(var_net_out)
        self.w_s = np.eye(self.dim_hidden_2, self.dim_input)

        # TODO: clarify whether we can initialte biases with zeros
        self.b1 = np.zeros((self.dim_hidden_1, 1))
        self.b2 = np.zeros((self.dim_hidden_2, 1))
        self.b_out = np.zeros((self.dim_output, 1))

    def _forward_pass(self, x):
        '''
        Performs forward pass of the network.

        a_i - results of applying weights for the data from precious layear
        z_i - result of activation

        We save the results for further backward pass.
        :param x: input data
        :return: multi-class prediction (n_class, 1)
        '''
        self.x = x
        self.a_1 = self.w1.dot(x) + self.b1
        self.z_1 = Tanh.activation(self.a_1)

        self.a_2 = self.w2.dot(self.z_1) + self.b2
        self.z_2 = Tanh.activation(self.a_2)
        self.z_2_with_skip_connection = self.z_2 + self.w_s.dot(x)

        self.a_3 = self.w_out.dot(self.z_2_with_skip_connection) + self.b_out
        self.y_pred = Softmax.activation(self.a_3)
        # self.y_pred.reshape((1,len(self.y_pred)))
        # self.y_pred = softmax(self.a_3)

    def _backward_pass(self, y_true):
        """
        Performs back propagation by calcularing local gradients and updates work.

        delta_out.shape = (dim_output, batch_size)
        delta_b3.shape = (dim_output, 1)

        delta_2.shape = (dim_hidden_2, batch_size)
        delta_b2.shape = (dim_hidden_2, 1)

        delta_1.shape = (dim_hidden_1, batch_size)
        delta_b1.shape = (dim_hidden_1, 1)

        :param y_true: ground truth labels of classes (n_classes,1)
        """
        self.y_true = y_true
        delta_out = self._delta_cross_entropy(y_true.T)
        delta_b_out = np.mean(delta_out, axis=1, keepdims=True)

        delta_2 = Relu.derivative(self.a_2) * self.w_out.T.dot(delta_out)
        delta_b2 = np.mean(delta_2, axis=1, keepdims=True)

        delta_1 = Tanh.derivative(self.a_1) * self.w2.T.dot(delta_2)
        delta_b1 = np.mean(delta_1, axis=1, keepdims=True)

        # update weights and biases
        self.w_out -= self.learning_rate * delta_out.dot(self.z_2_with_skip_connection.T)
        self.w2 -= self.learning_rate * delta_2.dot(self.z_1.T)
        self.w1 -= self.learning_rate * delta_1.dot(self.x.T)

        self.b_out -= self.learning_rate * delta_b_out
        self.b2 -= self.learning_rate * delta_b2
        self.b1 -= self.learning_rate * delta_b1

    def _delta_cross_entropy(self, y_true):
        delta = self.y_pred.copy()
        delta = delta - y_true
        #delta /= len(y_true)
        return delta

    # TODO: fix cross_entropy
    def _cross_entropy_loss(self, y_true):
        '''
        :param y_true: ground truth class labels (n_samples, 1)
        :return: cross entropy loss
        '''
        m = y_true.shape[0]
        p = Softmax.activation(self.y_pred)
        log_likelihood = -np.sum(np.log(p[y_true, :]))  # this gives something strange
        loss = np.sum(log_likelihood) / m
        return loss

    def fit(self, x_train, y_train, x_test, y_test, n_epochs=500):
        """
        Train neural network.
        :param x_train: (n_features, n_samples)
        :param y_train: (n_samples,1) (labels are not one hot encoded)
        :param x_test: test x data (n_features, n_samples)
        :param y_test: test y data (n_samples,1)
        :param n_epochs: number of epochs
        :return: train history dict
        """
        train_history = {'loss': [], 'accuracy': []}
        for epoch in tqdm.tqdm(range(n_epochs)):

            for i in range(x_train.shape[1] // self.batch_size):
                start_idx = i * self.batch_size
                end_idx = (i + 1) * self.batch_size
                self._forward_pass(x_train[:, start_idx:end_idx])
                # loss = self._cross_entropy_loss(y_train[start_idx:end_idx])
                loss = log_loss(y_train[start_idx:end_idx], self.y_pred.T)
                self._backward_pass(y_train[start_idx:end_idx])

            train_history['loss'].append(loss)
            train_history['accuracy'].append(get_accuracy(y_test, self.predict(x_test)))
            if epoch % 20 == 0:
                print("Epoch: {0} Loss: {1:.3f} Test acc: {2:.3f}".format(epoch, train_history['loss'][-1],
                                                                          train_history['accuracy'][-1]))
        return train_history

    # TODO: make prediction one hot encoded
    def predict(self, x):
        """
        Predict class labels for the given input.
        :param x: input data (n_features, n_samples)
        :return predicted labels (n_samples, n_classes)
        """
        self._forward_pass(x)
        predicted_classes = np.argmax(self.y_pred, axis=0)
        #prediction = np.zeros((self.y_pred.shape[0], x.shape[1]))
        #TODO: change hardcoded number of classes
        onehot_encoder = OneHotEncoder(n_values = 4, sparse=False)
        prediction = onehot_encoder.fit_transform(predicted_classes.reshape(len(predicted_classes), 1))
        # prediction[]
        #return np.expand_dims(np.argmax(self.y_pred, axis=0), axis=1)
        return prediction


def get_accuracy(true_values, prediction):
    #true_values == 1
    #return np.sum(true_values == prediction) / len(true_values)
    return np.sum(np.multiply(true_values, prediction)) / len(true_values)
