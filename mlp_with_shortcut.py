import numpy as np
from activation_functions import Tanh, Relu, Softmax
#TODO: test whether our activations work ok - compare with sklearn
from sklearn.neural_network._base import softmax, relu, tanh


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

#TODO: find out where to use batch size
        self.batch_size = batch_size
        self.learning_rate = learning_rate

#TODO: test whether we really have Var == 1 for networks
    def xavier_init(self):
        # followed the next explanation of Xavier initialization
        # https://www.quora.com/What-is-an-intuitive-explanation-of-the-Xavier-Initialization-for-Deep-Neural-Networks

        # specifying estimated Variance of networks on each layer according to  'rule of thumb'
        var_net_1 = 2. / (self.dim_input + self.dim_hidden_1)
        var_net_2 = 2. / (self.dim_hidden_1 + self.dim_hidden_2)
        var_net_out = 2. / (self.dim_hidden_2 + self.dim_output)

        self.w1 = np.random.randn(self.dim_hidden_1, self.dim_input)*np.sqrt(var_net_1)
        self.w2 = np.random.randn(self.dim_hidden_2, self.dim_hidden_1)*np.sqrt(var_net_2)
        self.w_out = np.random.randn(self.dim_output, self.dim_hidden_2)*np.sqrt(var_net_out)
        self.ws = np.eye(self.dim_hidden_2, self.dim_input)

        # TODO: clarify whether we can initialte biases with zeros
        self.b1 = np.zeros(self.dim_hidden_1)
        self.b2 = np.zeros(self.dim_hidden_2)
        self.b3 = np.zeros(self.dim_output)

    def forward_pass(self, x):
        '''
        Performs forward pass of the network.

        a_i - results of applying weights for the data from precious layear
        z_i - result of activation

        We save the results for further backprop.
        :param x: input data
        :return: multiclass prediction
        '''
        self.x = x
        self.a_1 = self.w1.dot(x) + self.b1
        self.z_1 = Tanh.activation(self.a_1)

        self.a_2 = self.w2.dot(self.z_1) + self.b2
        self.z_2 = Tanh.activation(self.a_2)
        self.z_2_with_skip_connection = self.z_2 + self.ws.dot(x)

        a_3 = self.w_out.dot(self.z_2_with_skip_connection) + self.b3
        self.y_pred = Softmax.activation(a_3)
