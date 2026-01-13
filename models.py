import numpy as np
from functions import sigmoid, softmax, identity, accuracy, cross_entropy_error, mean_square_error, numerical_gradient

class SimpleNet(object):
    
    def __init__(self):
        self.W = np.random.rand(2, 3)
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        return cross_entropy_error(y, t)
    
    def calc_gradient(self, x, t):
        f = lambda w: self.loss(x, t)
        W_grads = numerical_gradient(self.W, f)
        return W_grads

class SingleLayerNet(object):
    
    def __init__(self, num_input_node: int, num_w1_node: int, weight_init_std: float) -> None:
        self.W1 = weight_init_std * np.random.randn(num_input_node, num_w1_node)
        self.b1 = np.zeros(num_w1_node)

    def predict(self, x: np.ndarray) -> np.ndarray:
        a1 = np.dot(x, self.W1) + self.b1
        z = identity(a1)
        y = softmax(z)
        return y

    def accuracy(self, x: np.ndarray, t: np.ndarray) -> float:
        y = self.predict(x)
        t_predict = np.argmax(y, axis=1)
        acc = np.sum(t == t_predict) / t.shape[0]
        return round(acc, 4)
    
    def loss(self, x, t: np.ndarray) -> float:
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def gradient(self, x, t) -> tuple[np.ndarray, np.ndarray]:
        print('Calculating numerical gradient takes a while. Please be patient!')
        f = lambda W: self.loss(x, t)
        self.W1_grad = numerical_gradient(self.W1, f)
        self.b1_grad = numerical_gradient(self.b1, f)
        
        return self.W1_grad, self.b1_grad
    
    def gradient_descent(self, x, t, learning_rate, step_num):  
        for i in range(step_num):
            W1_grad_new, b1_grad_new = self.gradient(x, t)
            self.W1_grad -= learning_rate * W1_grad_new
            self.b1_grad -= learning_rate * b1_grad_new
        return self.W1_grad, self.b1_grad

class TwoLayerNet(object):
    
    def __init__(self, num_in_node, num_w1_node: int, num_w2_node: int):
        self.W1 = np.random.rand(num_in_node, num_w1_node)
        self.W2 = np.random.rand(num_w1_node, num_w2_node)
        self.b1 = np.ones(num_w1_node)
        self.b2 = np.ones(num_w2_node)
        self.params = {'W1': self.W1, 'W2': self.W2, 'b1': self.b1, 'b2': self.b2}
        print("An object has been intiallized from the TwoLayerNet class!")

    def predict(self, x):
        a1 = sigmoid(np.dot(x, self.W1) + self.b1)
        z = np.dot(a1, self.W2) + self.b2
        t = np.argmax(z, axis=1)
        return z, t

    def accuracy(self, x, t):
        _, t_predict = self.predict(x)
        acc = accuracy(t, t_predict)
        print("The accuracy of the prediction is: ", acc)
        return acc

    def loss(self, x, t):
        z, _ = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss
    
    def gradient(self, x, t):
        f = lambda W: self.loss(x, t)
        self.W1_grad = numerical_gradient(self.W1, f)
        self.W2_grad = numerical_gradient(self.W2, f)
        self.b1_grad = numerical_gradient(self.b1, f)
        self.b2_grad = numerical_gradient(self.b2, f)
        self.grads = {'W1_grad': self.W1_grad, 'W2_grad': self.W2_grad, 'b1_grad': self.b1_grad, 'b2_grad': self.b2_grad}
        print("Finish calculating the gradients!")

class ThreeLayerNet(object):
    
    def __init__(self, x_train, num_w1_node: int, num_w2_node: int, num_w3_node: int):
        self.x_train = x_train
        self.W1 = np.random.rand(self.x_train.shape[1], num_w1_node)
        self.W2 = np.random.rand(num_w1_node, num_w2_node)
        self.W3 = np.random.rand(num_w2_node, num_w3_node)
        self.b1 = np.ones(num_w1_node)
        self.b2 = np.ones(num_w2_node)
        self.b3 = np.ones(num_w3_node)
        print("An object has been intiallized from the ThreeLayerNet class!")

    def predict(self):
        self.a1 = np.dot(self.x_train, self.W1) + self.b1
        self.z1 = sigmoid(self.a1)
        self.a2 = np.dot(self.z1, self.W2) + self.b2
        self.z2 = sigmoid(self.a2)
        self.y = softmax(np.dot(self.z2, self.W3) + self.b3)
        self.t_predict = np.argmax(self.y, axis=1)
        print("The predicted labels are ready!")

    def calculate_accuracy(self, t_train):
        self.t_train = t_train
        self.accuracy = accuracy(self.t_train, self.t_predict)
        print("The accuracy of the prediction is: ", round(self.accuracy, 2))  

class Affine(object):

    def __init__(self, W: np.ndarray, b: np.ndarray) -> None:
        self.W = W
        self.b = b

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        z = np.dot(self.x, self.W) + self.b
        return z
    
    def backward(self, dout) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dx = np.dot(dout, self.W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)
        return dx, dW, db

class ReLU(object):

    def __init__(self) -> None:
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x<=0
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout[self.mask] = 0
        dx = dout
        return dx

class IdentityWithLoss(object):

    def __init__(self) -> None:
        pass

    def forward(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        self.y_pred = x
        self.y = y
        loss = mean_square_error(self.y_pred, self.y)
        return self.y_pred, loss

    def backward(self, dout=1) -> np.ndarray:
        dx = (2 / self.y.shape[0]) * (self.y_pred - self.y)
        return dx
    
class SoftmaxWithLoss(object):

    def __init__(self) -> None:
        pass

    def forward(self, x: np.ndarray, t):
        self.y = softmax(x)
        self.t = t
        loss = cross_entropy_error(self.y, self.t)
        return self.y, loss

    def backward(self, dout=1) -> np.ndarray:
        batch_size = self.y.shape[0]
        dx = dout * (self.y - self.t) / batch_size
        return dx

class DNN(object): # DNN and MLP are often used interchangeably

    def __init__(self, n_input_node, n_hidden_layer, n_hlayer_node, n_out_node, weight_init_std: float):
        self.config = {}
        self.config['n_hidden_layer'] = n_hidden_layer
        self.config['n_input_node'] = n_input_node
        self.config['n_out_node'] = n_out_node
        self.params = {}
        for i in range(n_hidden_layer+1):
            if i == 0:
                self.params['W'+str(i+1)] = np.random.randn(n_input_node, n_hlayer_node) * weight_init_std
                self.params['b'+str(i+1)] = np.zeros(n_hlayer_node)
            elif i == n_hidden_layer:
                self.params['W'+str(i+1)] = np.random.randn(n_hlayer_node, n_out_node) * weight_init_std
                self.params['b'+str(i+1)] = np.zeros(n_out_node)
            else:
                self.params['W'+str(i+1)] = np.random.randn(n_hlayer_node, n_hlayer_node) * weight_init_std
                self.params['b'+str(i+1)] = np.zeros(n_hlayer_node)

    def train(self, x: np.ndarray, y: np.ndarray, max_iter: int, learning_rate: float, batch_size: int):
        num_iter = 0
        mask = np.array([])
        while num_iter <= max_iter-1:
            num_iter += 1
            idx = np.random.choice(a=np.setdiff1d(np.arange(x.shape[0]), mask), size=batch_size)
            mask = np.append(mask, idx)
            x_batch = x[idx]
            y_batch = y[idx]
            # forward
            affine1 = Affine(self.params['W1'], self.params['b1'])
            affine2 = Affine(self.params['W2'], self.params['b2'])
            affine3 = Affine(self.params['W3'], self.params['b3'])
            affine4 = Affine(self.params['W4'], self.params['b4'])
            relu1 = ReLU()
            relu2 = ReLU()
            relu3 = ReLU()
            # identity_with_loss = IdentityWithLoss()
            softmax_with_loss = SoftmaxWithLoss()
            a1 = affine1.forward(x_batch)
            z1 = relu1.forward(a1)
            a2 = affine2.forward(z1)
            z2 = relu2.forward(a2)
            a3 = affine3.forward(z2)
            z3 = relu3.forward(a3)
            a4 = affine4.forward(z3)
            # _, loss = identity_with_loss.forward(a4, y)
            _, loss = softmax_with_loss.forward(a4, y_batch)

            # backward
            grads = {}
            dL = 1
            # da4 = identity_with_loss.backward(dL)
            da4 = softmax_with_loss.backward(dL)
            dz3, dW4, db4 = affine4.backward(da4)
            grads['W4'], grads['b4'] = dW4, db4
            da3 = relu3.backward(dz3)
            dz2, dW3, db3 = affine3.backward(da3)
            grads['W3'], grads['b3'] = dW3, db3
            da2 = relu2.backward(dz2)
            dz1, dW2, db2 = affine2.backward(da2)
            grads['W2'], grads['b2'] = dW2, db2
            da1 = relu1.backward(dz1)
            _, dW1, db1 = affine1.backward(da1)
            grads['W1'], grads['b1'] = dW1, db1

            # update loss
            _, loss = self.predict(x_batch, y_batch)

            # update parameters
            for key in self.params:
                self.params[key] -= learning_rate * grads[key]

    def predict(self, x: np.ndarray, y: np.ndarray):
        affine1 = Affine(self.params['W1'], self.params['b1'])
        affine2 = Affine(self.params['W2'], self.params['b2'])
        affine3 = Affine(self.params['W3'], self.params['b3'])
        affine4 = Affine(self.params['W4'], self.params['b4'])
        relu1 = ReLU()
        relu2 = ReLU()
        relu3 = ReLU()
        # identity_with_loss = IdentityWithLoss()
        softmax_with_loss = SoftmaxWithLoss()
        a1 = affine1.forward(x)
        z1 = relu1.forward(a1)
        a2 = affine2.forward(z1)
        z2 = relu2.forward(a2)
        a3 = affine3.forward(z2)
        z3 = relu3.forward(a3)
        a4 = affine4.forward(z3)
        # y_pred, loss = identity_with_loss.forward(a4, t)
        y_pred, loss = softmax_with_loss.forward(a4, y)
        return y_pred, loss