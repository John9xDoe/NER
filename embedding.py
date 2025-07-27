import numpy as np

class Embedding:
    def __init__(self, vocab_size, embedding_dim, lr=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weights = np.random.rand(vocab_size, embedding_dim) * 0.01
        self.lr = lr

        self.last_input = None

    def __call__(self, input_ids):
        self.last_input = input_ids
        return self.weights[input_ids]

    def backward(self, upstream_grad):
        grad = np.zeros_like(self.weights)

        for b in range(self.last_input.shape[0]):
            for t in range(self.last_input.shape[1]):
                idx = self.last_input[b, t]
                grad[idx] += upstream_grad[b, t]

        self.grad = grad

    def update(self):
        self.weights -= self.grad * self.lr

embedding = Embedding(vocab_size=10, embedding_dim=4)

tokens = [4, 2, 9]

vectors = embedding(tokens)

print(vectors)