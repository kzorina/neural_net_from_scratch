import numpy as np
import activation_functions


class Network:
    """
    Constructs a feed-forward multilayer perceptron with two hidden layers and
    ResNet-like shortcut connections.

    Architecture:
    output = softmax((ReLU(tanh(w1*input + b1)*w2 + b2) + ws*x)*w_out + b3)
    """
    def __init__(self, dim_input, dim_hidden_1, dim_hidden_2, dim_output, learning_rate, batch_size):
        self.dim_input = dim_input
        self.dim_hidden_1 = dim_hidden_1
        self.dim_hidden_2 = dim_hidden_2
        self.dim_output = dim_output
        self.xavier_init()

        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def xavier_init(self):
        # followed the next explanation of Xavier initialization
        # https://www.quora.com/What-is-an-intuitive-explanation-of-the-Xavier-Initialization-for-Deep-Neural-Networks
        var_net_1 = 1. / (self.dim_input + self.dim_hidden_1)
        var_net_2 = 1. / (self.dim_hidden_1 + self.dim_hidden_2)
        var_net_out = 1. / (self.dim_hidden_2 + self.dim_output)

        self.w1 = np.random.randn(self.dim_hidden_1, self.dim_input)*np.sqrt(var_net_1)
        self.w2 = np.random.randn(self.dim_hidden_2, self.dim_hidden_1)*np.sqrt(var_net_2)
        self.w_out = np.random.randn(self.dim_output, self.dim_hidden_2)*np.sqrt(var_net_out)
        self.ws = np.eye(self.dim_hidden_2, self.dim_input)

        # TODO: clarify whether we can initialte biases with zeros
        self.b1 = np.zeros(self.dim_hidden_1)
        self.b2 = np.zeros(self.dim_hidden_2)
        self.b3 = np.zeros(self.dim_hidden_out)
