import numpy as np


class Relu:
    @staticmethod
    def activation(x):
        x[x < 0] = 0
        return x

    @staticmethod
    def derivative(x):
        x[x < 0] = 0
        x[x > 0] = 1
        return x


class Tanh:
    @staticmethod
    def activation(x):
        return np.tanh(x)

    @staticmethod
    def derivative(x):
        return 1 - Tanh.activation(x) ** 2


class Softmax:
    @staticmethod
    def activation(x):
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=0)
