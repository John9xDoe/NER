import numpy as np

class Softmax:
    def forward(self, logits):
        logits -= np.max(logits, axis=-1, keepdims=True) # axis=-1 - последня размерность (обычно классы)
        exps = np.exp(logits)
        self.probs = exps / np.sum(exps, axis=-1, keepdims=True)
        return self.probs

    def backward(self, d_out):
        return d_out
