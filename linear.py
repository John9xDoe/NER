import numpy as np

class Linear:
    def __init__(self, input_dim, output_dim, lr=0.01):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr

        self.W = np.random.rand(input_dim, output_dim) * 0.01
        self.b = np.zeros(output_dim)

        self.last_input = None

    def __call__(self, x): # forward
        self.last_input = x
        return x @ self.W + self.b

    def backward(self, grad_output):
        print("grad_output shape:", grad_output.shape)
        print("self.output_dim:", self.output_dim)

        self.grad_W = self.last_input.reshape(-1, self.input_dim).T @ grad_output.reshape(-1, self.output_dim)
        self.grad_b = grad_output.sum(axis=(0, 1))

        return (grad_output @ self.W.T).reshape(self.last_input.shape)

    def update(self):
        self.W -= self.lr * self.grad_W
        self.b -= self.lr * self.grad_b

