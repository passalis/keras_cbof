from keras.datasets import mnist
import keras
import cv2
import numpy as np


def load_mnist():
    """
    Loads the MNIST dataset
    :return:
    """
    img_rows, img_cols = 28, 28
    num_classes = 10

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


def resize_images(x, scale=0.8):
    """
    Resizes a collection of grayscale images
    :param x: list of images
    :param scale: the scale
    :return:
    """
    img_size = [int(x * scale) for x in x.shape[1:]]
    new_data = np.zeros((x.shape[0], img_size[0], img_size[1], x.shape[3]))
    for k in range(x.shape[0]):
        new_data[k, :, :, 0] = cv2.resize(x[k][:, :, 0], (img_size[0], img_size[1]))
    return np.float32(new_data)
