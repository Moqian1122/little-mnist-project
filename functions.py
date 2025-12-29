import numpy as np

# activation functions

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

def identity(x):
    return x

# loss functions

def mean_square_error(label, prediction, batch_size):
    return (1/batch_size) * np.sum((label - prediction)**2)

def cross_entropy_error(label, y):
    if y.ndim == 1:
        y.reshape(1, y.size)
        label.reshape(1, label.size)

    if label.size == y.size:
        label = np.argmax(label, axis = 1)

    delta = 1e-7
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), label]) + delta) / batch_size

# accuracy functions

def accuracy(label, prediction):
    return round(np.sum([label == prediction]) / label.shape[0], 4)

# gradient functions (numeric(al) derivation applies.)

def gradient(x):
    grads = []
    h = 1e-4
    
    return grads