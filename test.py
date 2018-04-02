from deepautoencoder import StackedAutoEncoder
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

def test_data():
    print("Loading data files...")
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    data, target = mnist.train.images, mnist.train.labels

    # train / test  split
    idx = np.random.rand(data.shape[0]) < 0.8
    train_X, train_Y = data[idx], target[idx]
    test_X, test_Y = data[~idx], target[~idx]

    return train_X,train_Y,test_X,test_Y

def test_class(var):
    model = StackedAutoEncoder(dims=[200, 200], activations=['linear', 'linear'],noise=var, epoch=[
                               500, 500], loss='rmse', lr=0.01, batch_size=100, print_step=100, optimizer='adam')
    return model

def test_fit():
    model.fit(train_X)
    test_X_ = model.transform(test_X)
    return test_X_

if __name__ == '__main__':

    #Load the data
    train_X, train_Y, test_X, test_Y = test_data()

    print(type(train_X))

    model = test_class("gaussian")
    test_X_  = test_fit()

"""
    first_image = test_X_[500]
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28, 28))
    print(test_Y[500])
    plt.imshow(pixels, cmap='gray')
    plt.show()
"""


