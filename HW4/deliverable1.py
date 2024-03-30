import numpy as np
import matplotlib.pyplot as plt

A = C = np.array([[1.25, 0.75], [0.75, 1.25]])
b = d = np.array([0, 0])

def identity(x):
    return x

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def compute_rnn(x_0, A, b, C, d, activation, T=16):
    norms = np.zeros(T)
    x_t = x_0
    for t in range(T):
        x_t = activation(A @ x_t + b)
        y_t = C @ x_t + d
        norms[t] = np.linalg.norm(y_t)
    return norms


x_0_samples = np.random.randn(10, 2)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
activations = [identity, relu, tanh]
activation_names = ['Identity', 'ReLU', 'Tanh']


for i, (activation, name) in enumerate(zip(activations, activation_names)):
    for x_0 in x_0_samples:
        norms = compute_rnn(x_0, A, b, C, d, activation)
        axes[i].plot(range(16), norms, label=f'x_0: {x_0}')
    axes[i].set_title(f'Activation: {name}')
    axes[i].set_xlabel('t')
    axes[i].set_ylabel('||y_t||2')

plt.tight_layout()
plt.savefig("del1.png")




x_0_conditions = [np.array([1, 1]), np.array([1, -1])]
idx = 0
for x_0 in x_0_conditions:
    for activation, name in zip(activations, activation_names):
        plt.figure(figsize=(10, 6))
        norms = compute_rnn(x_0, A, b, C, d, activation)
        plt.plot(range(16), norms, label=f'x_0: {x_0}, {name}')
        plt.title('Relationship between t and ||y_t||2 for x_0 = [1, 1] and x_0 = [1, -1]')
        plt.xlabel('t')
        plt.ylabel(f'{name}, {x_0}')
        plt.savefig(f"del_{idx}.png")
        idx += 1

plt.title('Relationship between t and ||y_t||2 for x_0 = [1, 1] and x_0 = [1, -1]')
plt.xlabel('t')
plt.ylabel('||y_t||2')
plt.legend()
plt.savefig("del1_1.png")

