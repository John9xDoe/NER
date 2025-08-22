import numpy as np

class CrossEntropy:
    def forward(self, probs, y_true):

        self.probs = np.array(probs)
        self.y_true = np.array(y_true)

        B, T, C = self.probs.shape

        print("probs shape:", probs.shape)
        print("y_true shape:", np.array(y_true).shape)

        flat_probs = probs.reshape(-1, C)
        flat_labels = np.array(y_true).reshape(-1)

        print("flat_labels:", flat_labels)
        print("max label:", flat_labels.max())
        print("min label:", flat_labels.min())

        print("flat_probs shape:", flat_probs.shape)
        print("flat_labels shape:", flat_labels.shape)

        correct_logprobs = -np.log(flat_probs[np.arange(len(flat_labels)), flat_labels] + 1e-9)
        return np.mean(correct_logprobs)

    def backward(self):
        B, T, C = self.probs.shape

        flat_probs = self.probs.reshape(-1, C).copy()
        flat_labels = np.array(self.y_true).reshape(-1)

        grad = flat_probs
        grad[np.arange(len(flat_labels)), flat_labels] -= 1
        grad /= B * T
        return grad.reshape(B, T, C)