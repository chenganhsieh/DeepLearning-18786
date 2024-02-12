import numpy as np
# Loss functions
def l2_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def l2_loss_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size

def cross_entropy_loss(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9)) / y_true.shape[0]

def cross_entropy_derivative(y_pred, y_true):
    return y_pred - y_true
