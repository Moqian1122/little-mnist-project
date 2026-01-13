import numpy as np

# activation functions

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x - np.max(x, axis=-1, keepdims=True)) / np.sum(np.exp(x - np.max(x, axis=-1, keepdims=True)), axis=-1, keepdims=True)

def identity(x):
    return x

# from class labels to one-hot representation

def one_hot_encoding(x: np.ndarray, n_out_node) -> np.ndarray:
    y = np.zeros(shape=(x.size, n_out_node))
    y[np.arange(x.size), x] = 1
    return y

# training and validaton sets splitting

def train_val_split(x: np.ndarray, y: np. ndarray, split_ratio: float):
    '''
    split_ratio: the ratio of the size of set validation set to the size of the training set
    '''
    mask = np.random.choice(a=x.shape[0], size=int(x.shape[0]*split_ratio), replace=False)
    x_val, y_val = x[mask], y[mask]
    x_train, y_train = x[np.setdiff1d(np.arange(x.shape[0]), mask)], y[np.setdiff1d(np.arange(x.shape[0]), mask)]
    return x_train, y_train, x_val, y_val 

# loss functions

def mean_square_error(label, prediction):
    batch_size = label.shape[0]
    return (1/batch_size) * np.sum((label - prediction)**2)

def cross_entropy_error(y, t) -> np.ndarray:
    if y.ndim == 1:
        y = y.reshape(1, y.size)
    if t.ndim == 1:
        t = t.reshape(1, t.size)
    if y.ndim == t.ndim:
        t = np.argmax(t, axis=1)
    delta = 1e-7
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(y.shape[0]), t] + delta)) / batch_size

# accuracy functions

def accuracy(label, prediction):
    return round(np.sum([label == prediction]) / label.shape[0], 4)

# gradient functions (numeric(al) derivation applies.)

def numerical_gradient(x: np.ndarray, f):
    
    grad = np.zeros_like(x)
    h = 1e-4 # 0.0001
    
    it = np.nditer(op=x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:

        idx = it.multi_index

        tmp = x[idx]
        x[idx] = tmp + h
        y1 = f(x)
        x[idx] = tmp - h
        y2 = f(x)
        grad_val = (y1 - y2) / (2*h)
        grad[idx] = grad_val
        x[idx] = tmp

        it.iternext()

    return grad

def gradient_descent(f, x, learning_rate, step_num):
    
    for i in range(step_num):
        grad = numerical_gradient(x, f)
        x -= learning_rate * grad

    return x