from mlp import MLP, train_mlp, test_mlp
from activation import relu, sigmoid, tanh, linear
from activation import relu_derivative, sigmoid_derivative, tanh_derivative, linear_derivative
from numpyNN import sample_data, plot_loss, plot_decision_boundary
from collections import defaultdict
import random
import numpy as np

random.seed(0)
np.random.seed(0)

epochs = 28000
learning_rate = 1e-4

logs = defaultdict(list)
x_train, y_train, x_test, y_test = sample_data(data_name='sinusoid')

layer_sizes = [2, 10, 8, 1]
activation_funcs = [relu, relu, sigmoid]
activation_derivatives_funcs = [relu_derivative, relu_derivative, sigmoid_derivative]
mlp = MLP(layer_sizes, activation_funcs, activation_derivatives_funcs,"xavier")

train_mlp(mlp=mlp, training_data=[x_train, y_train], num_epoch=epochs, 
opt_loss='cross_entropy', opt_optim='Adam', learning_rate=learning_rate, logs=logs, val_data = [x_test, y_test])

plot_loss(logs)
plot_decision_boundary(x_train,y_train,mlp.forward, None, "sinusoid_adam.png", epochs, learning_rate, logs)



