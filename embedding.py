import numpy as np

class Embedding:
    def __init__(self, vocab_size, embedding_dim, lr):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weights = np.random.rand(vocab_size, embedding_dim) * 0.01
        self.lr = lr

    def __call__(self, input_ids):
        self.last_input = input_ids
        return self.weights[input_ids]

    def backward(self, upstream_grad):

        self.grad = []

        for b in range(self.last_input.shape[0]):
            for t in range(self.last_input.shape[1]):
                idx = self.last_input[b, t]
                self.grad[idx] += upstream_grad[b, t]

    def upgrade(self):
        self.weights -= self.lr * self.grad


embedding = Embedding(vocab_size=4, embedding_dim=4, lr=0.01)

tokens = [0, 1, 2, 3]

print(embedding(tokens))