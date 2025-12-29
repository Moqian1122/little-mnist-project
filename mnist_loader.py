import os
import numpy as np

def _load_img(file_name):
    dataset_dir = os.getcwd() + '/mnistdata'
    file_path = dataset_dir + "/" + file_name
    img_size = 784

    print("Converting " + file_name + " to NumPy Array ...")
    with open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")

    return data

def _load_label(file_name):
    dataset_dir = os.getcwd() + '/mnistdata'
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels

def load_mnist():

    key_file = {
    'train_img':'train-images.idx3-ubyte',
    'train_label':'train-labels.idx1-ubyte',
    'test_img':'t10k-images.idx3-ubyte',
    'test_label':'t10k-labels.idx1-ubyte'
    }

    print("Just wait a sec. The data is loading...\n")

    x_train = _load_img(key_file["train_img"])
    x_test = _load_img(key_file["test_img"])
    t_train = _load_label(key_file["train_label"])
    t_test = _load_label(key_file["test_label"])

    print("Data has been successfully loaded!\n")

    return x_train, x_test, t_train, t_test