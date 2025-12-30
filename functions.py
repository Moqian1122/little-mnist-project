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

def mean_square_error(label, prediction):
    batch_size = label.shape[0]
    return (1/batch_size) * np.sum((label - prediction)**2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    if y.size == t.size:
        t = np.argmax(t, axis = 1)

    delta = 1e-7
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t]) + delta) / batch_size

# accuracy functions

def accuracy(label, prediction):
    return round(np.sum([label == prediction]) / label.shape[0], 4)

# gradient functions (numeric(al) derivation applies.)

def numerical_gradient(x: np.ndarray, f):
    grads = np.zeros_like(x)
    h = 1e-4
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:

        idx = it.multi_index

        tmp = x[idx]
        x[idx] = tmp + h
        y1 = f(x)
        x[idx] = tmp - h
        y2 = f(x)
        grad = (y1 - y2) / (2*h)
        grads[idx] = grad
        x[idx] = tmp

        it.iternext()

    return grads

def gradient_descent():
    pass