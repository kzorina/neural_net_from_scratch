import numpy as np

# TODO: check whether this work properly
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
        return 1 - Tanh.activation(x)**2

# TODO: decide whether we need softmax derivative here
class Softmax:
    @staticmethod
    def activation(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    # @staticmethod
    # def derivative(z):
    #     return Sigmoid.activation(z) * (1 - Sigmoid.activation(z))





