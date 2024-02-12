from mlp import MLP, train_mlp, test_mlp
from activation import relu, sigmoid, tanh, linear
from activation import relu_derivative, sigmoid_derivative, tanh_derivative, linear_derivative
from numpyNN import sample_data, plot_loss_nonlinear, plot_decision_boundary_3d
from collections import defaultdict
import random
import numpy as np

random.seed(0)
np.random.seed(0)

epochs = 41000
learning_rate = 1e-4
target = 'swiss-roll' # 'xor' 'swiss-roll'

logs = defaultdict(list)
x_train, y_train, x_test, y_test = sample_data(data_name=target)
if target == 'circle':
    new_feature = (x_train[:, 0] ** 2 + x_train[:, 1] ** 2).reshape(-1, 1)
    x_train = np.hstack((x_train, new_feature))
    new_feature = (x_test[:, 0] ** 2 + x_test[:, 1] ** 2).reshape(-1, 1)
    x_test = np.hstack((x_test, new_feature))
elif target == 'XOR':
    new_feature = (x_train[:, 0] * x_train[:, 1]).reshape(-1, 1)
    x_train = np.hstack((x_train, new_feature))
    new_feature = (x_test[:, 0] * x_test[:, 1]).reshape(-1, 1)
    x_test = np.hstack((x_test, new_feature))
elif target == 'swiss-roll':
    new_feature = (np.sign(x_train[:, 0]) * np.sqrt(x_train[:, 0] ** 2 + x_train[:, 1] ** 2)).reshape(-1, 1)
    x_train = np.hstack((x_train, new_feature))
    new_feature = (np.sign(x_test[:, 0]) * np.sqrt(x_test[:, 0] ** 2 + x_test[:, 1] ** 2)).reshape(-1, 1)
    x_test = np.hstack((x_test, new_feature))

layer_sizes = [3, 10, 8, 1] # 10 8
activation_funcs = [relu, relu, sigmoid]
activation_derivatives_funcs = [relu_derivative, relu_derivative, sigmoid_derivative]
mlp = MLP(layer_sizes, activation_funcs, activation_derivatives_funcs,"he")

train_mlp(mlp=mlp, training_data=[x_train, y_train], num_epoch=epochs, 
opt_loss='cross_entropy', opt_optim='Adam', learning_rate=learning_rate, logs=logs, val_data = [x_test, y_test])

plot_loss_nonlinear(logs)
plot_decision_boundary_3d(x_train,y_train,mlp.forward, None, "nonlinear_"+target+".png", epochs, learning_rate, logs, target)



