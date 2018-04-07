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


def model_def(var):
    model = StackedAutoEncoder(dims=[500], activations=['linear'],noise=var, epoch=[
                               2000], loss='rmse', lr=0.01, batch_size=100, print_step=100, optimizer='adam',
                               graph=True)
    return model


def model_eval(model, test_X, noise=True):
    if noise:
        test_In = model.add_noise(test_X)
    else:
        test_In = test_X

    test_X_ = model.transform(test_In)

    #mean squared error for all examples
    mse = ((test_X - test_X_) **2).mean(axis=1)
    amse = mse.mean()

    return test_X_, mse


def model_plot(test, fit_noised, transform_noised, test_, num):

    original_image = test[num]
    original_image = np.array(original_image, dtype='float')
    pixels1 = original_image.reshape((28, 28))

    noised_image = fit_noised[num]
    noised_image = np.array(noised_image, dtype='float')
    pixels2 = noised_image.reshape((28, 28))

    # Pixel 2 and 3 should be similar, because testing with training set

    noised_image = transform_noised[num]
    noised_image = np.array(noised_image, dtype='float')
    pixels3 = noised_image.reshape((28, 28))

    denoised_image = test_[num]
    denoised_image = np.array(denoised_image, dtype='float')
    pixels4 = denoised_image.reshape((28, 28))

    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(pixels1, cmap='gray')
    axarr[1].imshow(pixels2, cmap='gray')
    axarr[2].imshow(pixels3, cmap='gray')
    axarr[3].imshow(pixels4, cmap='gray')


if __name__ == '__main__':


    #Load the data
    train_X, train_Y, test_X, test_Y = test_data()
    #define model
    model = model_def("gaussian")
    #fit data
    model.fit(train_X)
    #transform data
    train_X_, mse = model_eval(model, train_X, noise=True)

    #plot data
    model_plot(train_X, model.fit_noised, model.transform_noised, train_X_, 200)

    #plt.plot(model.loss_history[0],'x')
    #plt.show()


    #first_image = test_X_[500]
    #first_image = np.array(first_image, dtype='float')
    #pixels = first_image.reshape((28, 28))
    #print(test_Y[500])
    #plt.imshow(pixels, cmap='gray')
    #plt.show()

    #Visualize the noised and theoretically denoised data

    """
    original_image = train_X[500]
    original_image = np.array(original_image, dtype='float')
    pixels1 = original_image.reshape((28, 28))

    noised_image = model.fit_noised[500]
    noised_image = np.array(noised_image, dtype='float')
    pixels2 = noised_image.reshape((28, 28))
    
    # Pixel 2 and 3 should be the same, because testing with training set
    
    noised_image = model.transform_noised[500]
    noised_image = np.array(noised_image, dtype='float')
    pixels3 = noised_image.reshape((28, 28))

    denoised_image = train_X_[500]
    denoised_image = np.array(denoised_image, dtype='float')
    pixels4 = denoised_image.reshape((28, 28))

    f, axarr = plt.subplots(1,4)
    axarr[0].imshow(pixels1, cmap='gray')
    axarr[1].imshow(pixels2, cmap='gray')
    axarr[2].imshow(pixels3, cmap='gray')
    axarr[3].imshow(pixels4, cmap='gray')
    """