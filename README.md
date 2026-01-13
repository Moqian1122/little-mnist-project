This is a repository for a deep learning project of neural networks development with MNIST.

**MNIST database**

The MNIST database was intialized by National Institute of Standards and Technology. In fact, MNIST is exactly an acronym for Modified National Institute of Standards and Technology. The offial site can be referred to as this [link](https://yann.lecun.org/exdb/mnist/).

The MNIST contains handwritten digits 0-9 collected from American high schools. It provides a baseline to test image processing systems. In this project, neural networks are built from scracth, after which they are trained, validated and tested on MNIST.

**Model Construction**

I built neural networks from scratch. The expression 'from scratch' means the process is free of machine learning frameworks like Tensorflow, Keras and Scikit-learn. I only used Numpy as it's an API allowing us to leverage C's low-level speed to help vectorized operation in Python codes.

First of all, I built DNN (Dense Neural Networks, or MLP, Multi-Layer Perceptron) with 3 hidden layers. After training, a specification of 3 hidden layers with 256 neruons each layer already achieves an accuracy of above 99% and the accuracy on the validation set is also 90%, suggesting no serious overfitting.

Afterwards, manual CNN (Convolutional Neural Networks) will be also constructed to compare the performance.