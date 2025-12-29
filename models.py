import numpy as np
from functions import sigmoid, softmax, identity, accuracy

class SingleLayerNet(object):
    
    def __init__(self, x_train, num_w1_node: int):
        self.x_train = x_train
        self.W1 = np.random.rand(x_train.shape[1], num_w1_node)
        self.b1 = np.ones(num_w1_node)
        print("An object has been intiallized from the SingleLayerNet class!")

    def predict(self):
        a1 = np.dot(self.x_train, self.W1) + self.b1
        y = identity(a1)
        self.t_predict = np.argmax(y, axis=1)
        print("The predicted labels are ready!")

    def calculate_accuracy(self, t_train):
        self.t_train = t_train
        self.accuracy = accuracy(self.t_train, self.t_predict)
        print("The accuracy of the prediction is: ", round(self.accuracy, 2))

class TwoLayerNet(object):
    
    def __init__(self, x_train, num_w1_node: int, num_w2_node: int):
        self.x_train = x_train
        self.W1 = np.random.rand(self.x_train.shape[1], num_w1_node)
        self.W2 = np.random.rand(num_w1_node, num_w2_node)
        self.b1 = np.ones(num_w1_node)
        self.b2 = np.ones(num_w2_node)
        print("An object has been intiallized from the TwoLayerNet class!")

    def predict(self):
        a1 = sigmoid(np.dot(self.x_train, self.W1) + self.b1)
        self.y = softmax(np.dot(a1, self.W2) + self.b2)
        self.t_predict = np.argmax(self.y, axis=1)
        print("The predicted labels are ready!")

    def calculate_accuracy(self, t_train):
        self.t_train = t_train
        self.accuracy = accuracy(self.t_train, self.t_predict)
        print("The accuracy of the prediction is: ", self.accuracy)

    def calculate_loss(self):
        pass

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

class FourLayerNet(object):
    
    def __init__(self, num_w1_node: int, num_w2_node: int, num_w3_node:int, num_w4_node: int):
        self.W1 = np.random.rand(num_w1_node)
        self.W2 = np.random.rand(num_w2_node)
        self.W3 = np.random.rand(num_w3_node)
        self.W4 = np.random.rand(num_w4_node)

class FiveLayerNet(object):
    
    def __init__(self, x_train, num_w1_node: int, num_w2_node: int, num_w3_node: int, num_w4_node: int, num_w5_node: int):
        self.x_train = x_train
        self.W1 = np.random.rand(self.x_train.shape[1], num_w1_node)
        self.W2 = np.random.rand(num_w1_node, num_w2_node)
        self.W3 = np.random.rand(num_w2_node, num_w3_node)
        self.W4 = np.random.rand(num_w3_node, num_w4_node)
        self.W5 = np.random.rand(num_w4_node, num_w5_node)
        self.b1 = np.ones(num_w1_node)
        self.b2 = np.ones(num_w2_node)
        self.b3 = np.ones(num_w3_node)
        self.b4 = np.ones(num_w4_node)
        self.b5 = np.ones(num_w5_node)
        print("An object has been intiallized from the ThreeLayerNet class!")

    def predict(self):
        self.a1 = sigmoid(np.dot(self.x_train, self.W1) + self.b1)
        self.a2 = sigmoid(np.dot(self.a1, self.W2) + self.b2)
        self.a3 = sigmoid(np.dot(self.a2, self.W3) + self.b3)
        self.a4 = sigmoid(np.dot(self.a3, self.W4) + self.b4)
        self.y = softmax(np.dot(self.a4, self.W5) + self.b5)
        self.t_predict = np.argmax(self.y, axis=1)
        print("The predicted labels are ready!")

    def calculate_accuracy(self, t_train):
        self.t_train = t_train
        self.accuracy = accuracy(self.t_train, self.t_predict)
        print("The accuracy of the prediction is: ", self.accuracy)  