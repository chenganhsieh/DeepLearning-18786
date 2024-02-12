from mlp import MLP, train_mlp, test_mlp
from activation import relu, sigmoid, tanh, linear
from activation import relu_derivative, sigmoid_derivative, tanh_derivative, linear_derivative
from numpyNN import sample_data, plot_loss, plot_decision_boundary
from collections import defaultdict
import random
import numpy as np

random.seed(0)
np.random.seed(0)

epochs = 2000
learning_rate = 1000

logs = defaultdict(list)
x_train, y_train, x_test, y_test = sample_data(data_name='XOR')

layer_sizes = [2, 5, 1]
activation_funcs = [sigmoid, sigmoid]
activation_derivatives_funcs = [sigmoid_derivative, sigmoid_derivative]
mlp = MLP(layer_sizes, activation_funcs, activation_derivatives_funcs,"xavier")

train_mlp(mlp=mlp, training_data=[x_train, y_train], num_epoch=epochs, 
opt_loss='l2', opt_optim='GD', learning_rate=learning_rate, logs=logs, val_data = [x_test, y_test])

plot_loss(logs)
plot_decision_boundary(x_train,y_train,mlp.forward, None, "xor.png", epochs, learning_rate, logs)



