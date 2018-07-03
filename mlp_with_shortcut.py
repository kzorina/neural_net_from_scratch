import numpy as np
import activation_functions


class Network:
    """
    Constructs a feed-forward multilayer perceptron with two hidden layers and
    ResNet-like shortcut connections.

    Architecture:
    output = softmax((ReLU(tanh(w1*input + b1)*w2 + b2) + ws*x)*w3 + b3)
    """
    def __init__(self, dimensions, learning_rate, batch_size):

        self.batch_size = batch_size
        self.learning_rate = learning_rate

