import numpy as np

class CrossEntropy:
    def forward(self, y_pred, y_true):
        return -np.log(y_true)
    def backward(self):
        pass

