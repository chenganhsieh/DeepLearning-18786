import numpy as np
from loss import l2_loss_derivative, cross_entropy_derivative
from loss import l2_loss, cross_entropy_loss

# MLP class
class MLP:
	def __init__(self, layer_sizes, activation_funcs, activation_derivatives_funcs, init_method):
		self.weights = []
		self.biases = []
		self.layers = len(layer_sizes)
		self.activation_funcs = activation_funcs
		self.activation_derivatives_funcs = activation_derivatives_funcs
		self.init_weights_and_biases(layer_sizes, init_method)

		# For Gradient descent with momentum
		self.velocity_weights = [np.zeros_like(w) for w in self.weights]
		self.velocity_biases = [np.zeros_like(b) for b in self.biases]

		# For Adam
		self.m_weights = [np.zeros_like(w) for w in self.weights]
		self.v_weights = [np.zeros_like(w) for w in self.weights]
		self.m_biases = [np.zeros_like(w) for w in self.biases]
		self.v_biases = [np.zeros_like(w) for w in self.biases]

	def init_weights_and_biases(self, layer_sizes, init_method):
		for i in range(1, len(layer_sizes)):
			if init_method == "xavier": # good for sigmoid
				limit = np.sqrt(6 / (layer_sizes[i - 1] + layer_sizes[i]))
				self.weights.append(np.random.uniform(-limit, limit, (layer_sizes[i - 1], layer_sizes[i])))
			elif init_method == "he": # good for ReLU
				limit = np.sqrt(2 / layer_sizes[i - 1])
				self.weights.append(np.random.normal(0, limit, (layer_sizes[i - 1], layer_sizes[i])))
			self.biases.append(np.zeros(layer_sizes[i]))

	def forward(self, X):
		input_x = X
		cache = {'A0': X}
		
		for i in range(1, self.layers):
			z = input_x @ self.weights[i - 1] + self.biases[i - 1]
			input_x = self.activation_funcs[i - 1](z)

			cache[f'Z{i}'] = z
			cache[f'A{i}'] = input_x

		return input_x, cache

	def backward(self, y_pred, y_true, cache, opt_loss):
		gradients = {}
		m = y_true.shape[0]

		# Derivative of the loss function
		if opt_loss == "l2":
			da = l2_loss_derivative(y_pred, y_true)
		elif opt_loss == "cross_entropy":
			da = cross_entropy_derivative(y_pred, y_true)

		for i in range(self.layers-1, 0, -1):
			dz = da * self.activation_derivatives_funcs[i - 1](cache[f'Z{i}'])
			dw = cache[f'A{i - 1}'].T @ dz / m
			import pdb
			pdb.set_trace()
			db = np.sum(dz, axis=0) / m
			da = dz @ self.weights[i - 1].T

			gradients[f'dW{i}'] = dw
			gradients[f'db{i}'] = db

		return gradients

	def update_parameters(self, gradients, learning_rate, opt_optim,  t, beta1 = 0.9, beta2 = 0.999, epilson=1e-9):
		if opt_optim == "GD":
			for i in range(1, self.layers):
				self.weights[i - 1] -= learning_rate * gradients[f'dW{i}']
				self.biases[i - 1] -= learning_rate * gradients[f'db{i}']
		elif opt_optim == "GDM":
			for i in range(1, self.layers):
				self.velocity_weights[i - 1] = beta1*self.velocity_weights[i - 1] + (1-beta1)*gradients[f'dW{i}']
				self.velocity_biases[i - 1] = beta1*self.velocity_biases[i - 1] + (1-beta1)*gradients[f'db{i}']

				self.weights[i - 1] -= learning_rate * self.velocity_weights[i - 1]
				self.biases[i - 1] -= learning_rate * self.velocity_biases[i - 1]
		elif opt_optim == "Adam":
			for i in range(1, self.layers):
				self.m_weights[i - 1] = beta1 * self.m_weights[i - 1] + (1 - beta1) * gradients[f'dW{i}']
				self.v_weights[i - 1] = beta2 * self.v_weights[i - 1] + (1 - beta2) * (gradients[f'dW{i}'] ** 2)

				self.m_biases[i - 1] = beta1 * self.m_biases[i - 1] + (1 - beta1) * gradients[f'db{i}']
				self.v_biases[i - 1] = beta2 * self.v_biases[i - 1] + (1 - beta2) * (gradients[f'db{i}'] ** 2)

				
				m_hat_weights = self.m_weights[i - 1] / (1 - beta1 ** (t+1))
				v_hat_weights = self.v_weights[i - 1] / (1 - beta2 ** (t+1))

				m_hat_biases = self.m_biases[i - 1] / (1 - beta1 ** (t+1))
				v_hat_biases = self.v_biases[i - 1] / (1 - beta2 ** (t+1))

				self.weights[i - 1] -= learning_rate * m_hat_weights / (np.sqrt(v_hat_weights + epilson))
				self.biases[i - 1] -= learning_rate * m_hat_biases / (np.sqrt(v_hat_biases + epilson))
			

def train_mlp(mlp, training_data, num_epoch, opt_loss, opt_optim, learning_rate, logs, val_data = None):
	for epoch in range(num_epoch):
		x, y_true = training_data
		y_pred, cache = mlp.forward(x)
		if opt_loss == "l2":
			train_loss = l2_loss(y_pred, y_true)
		elif opt_loss == "cross_entropy":
			train_loss = cross_entropy_loss(y_pred, y_true)
		gradients = mlp.backward(y_pred, y_true, cache, opt_loss)
		mlp.update_parameters(gradients, learning_rate, opt_optim, epoch)
		logs['train_loss'].append(train_loss)

		val_loss = 0
		if val_data:
			x, y_true = val_data
			y_pred, _ = mlp.forward(x)
			if opt_loss == "l2":
				val_loss = l2_loss(y_pred, y_true)
			elif opt_loss == "cross_entropy":
				val_loss = cross_entropy_loss(y_pred, y_true)
			# val_loss = val_loss / len(val_data)
			logs['val_loss'].append(val_loss)

		print(f"Epoch{epoch}: Train Loss:{train_loss}, Val Loss:{val_loss}")

def test_mlp(mlp, test_data, opt_loss, logs):
	total_loss = 0
	for x_batch, y_batch in test_data:
		y_pred, _ = mlp.forward(x_batch)
		if opt_loss == "l2":
			total_loss += l2_loss(y_pred, y_batch)
		elif opt_loss == "cross_entropy":
			total_loss += cross_entropy_loss(y_pred, y_batch)

	test_loss = total_loss / len(test_data)
	logs['test_loss'].append(test_loss)
