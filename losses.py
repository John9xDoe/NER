import numpy as np

class CrossEntropy:
    def forward(self, probs, y_true):
        self.probs = probs
        self.y_true = y_true
        batch_size = y_true.shape[0]

        correct_logprobs = -np.log(probs[np.arange(batch_size), y_true])
        return np.mean(correct_logprobs)

    def backward(self):
        batch_size = self.y_true.shape[0]
        grad = self.probs.copy[0]
        grad[np.arange(batch_size), self.y_true] -= 1
        grad /= batch_size
        return grad
