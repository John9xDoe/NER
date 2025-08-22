'''
import numpy as np

probs = np.array([
    [0.1, 0.6, 0.3],
    [0.7, 0.2, 0.1],
    [0.2, 0.2, 0.6]
])

y_true = np.array([1, 0, 2])

y_pred = probs[np.arange(len(y_true)), y_true]
print(y_pred)

log_probs = -np.log(y_pred + 1e-9)
print(log_probs)
loss = np.mean(log_probs)
print(loss)

logits = np.array([
    [1.0, 2.0, 3.0],
    [1.0, 2.0, -1.0]
])

a = logits - np.max(logits, axis=1, keepdims=True)

exps = np.exp(a)
print(exps)
probs = exps / np.sum(exps, axis=1, keepdims=True)

print(probs)

class Softmax:
    def forward(self, logits):
        logits -= np.max(logits, axis=1, keepdims=True)
        exps = np.exp(logits)
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)

        return self.probs

print()

logits = np.array([
    [2.0, 1.0, 0.1],
    [0.5, 2.5, 0.3]
])

y_true = np.array([0, 1])

logits -= np.max(logits, axis=1, keepdims=True)
exps = np.exp(logits)
probs = exps / np.sum(exps, axis=1, keepdims=True)

y_pred = probs[np.arange(len(y_true)), y_true]

loss = -np.mean(np.log(y_pred))

pred_classes = np.argmax(probs, axis=1)
accuracy = np.mean(y_true == pred_classes)
print(accuracy)

print('\n' * 10)

def forward(logits, y_true):

    logits -= np.max(logits, axis=1, keepdims=True)
    exps = np.exp(logits)
    probs = exps / np.sum(exps, axis=1, keepdims=True)

    y_pred = probs[np.arange(len(y_true)), y_true]
    loss = -np.mean(np.log(y_pred + 1e-9))
    pred_classes = np.argmax(probs, axis=1)
    accuracy = np.mean(y_true == pred_classes)

    return loss, probs, accuracy

print(forward(np.array([[2.0, 1.0, 0.1],[0.5, 2.5, 0.3]]), np.array([0, 1])))
'''