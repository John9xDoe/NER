from activations import Softmax
from embedding import Embedding
from linear import Linear

import numpy as np

from losses import CrossEntropy


class NER:
    def __init__(self, vocab_size, output_dim, embedding_dim, lr, label2id, id2label):
        self.embedding = Embedding(vocab_size, embedding_dim, lr)
        self.linear = Linear(embedding_dim, output_dim, lr)
        self.softmax = Softmax()
        self.loss_fn = CrossEntropy()

        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.lr = lr

        self.vocab = {}

        self.label2id = label2id
        self.id2label = id2label

    def forward(self, tokens):
        self.embeds = self.embedding(tokens)
        self.logits = self.linear(self.embeds)
        self.probs = self.softmax.forward(self.logits)

        return self.probs

    def predict(self, tokens):
        probs = self.forward(tokens)
        return np.argmax(probs, axis=1)

    def compute_loss(self, probs, y_true):
        return self.loss_fn.forward(probs, y_true)

    def backward(self, true_labels):
        grad_logits = self.softmax.backward(true_labels)
        grad_embds = self.linear.backward(grad_logits)
        self.embedding.backward(grad_embds)

        #return grad

    def train_step(self, tokens, y_true):
        probs = self.forward(tokens)
        loss = self.compute_loss(probs, y_true)
        self.backward(y_true)
        return loss





