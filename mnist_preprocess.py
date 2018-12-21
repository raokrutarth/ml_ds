
import numpy as np
from collections import namedtuple
import keras
from keras.datasets import mnist


Data = namedtuple('Data', ['x', 'y'])

class MINSTData:
    '''
        Preprocesses the keras MNIST dataset to be used
        in a neural network
    '''
    def __init__(self):
        np.random.seed(42)
        (x_train, y_train), (x_test, y_test) = mnist.load_data(path='./data/mnist.npz')
        self.train_data = Data(x_train, y_train)
        self.test_data = Data(x_test, y_test)
        self.num_test, self.num_train, self.image_width, self.image_height = (
            self.test_data.x.shape[0],
            self.train_data.x.shape[0],
            self.train_data.x.shape[1],
            self.train_data.x.shape[2])
        self.num_pixels = self.image_width * self.image_height

    def get_dimensions(self):
        '''
            Each datapoint in x is a 28x28 matrix representing
            a 28x28 pixel image.

            return: num_tests, num_train, num_pixels
        '''
        print('[+] Training data dimensions')
        print('[+] Training dataset (no. of datapoints, pixels, pixels): {}'.format(
            self.train_data.x.shape))
        print('[+] Training dataset labels (no. of datapoints, ): {}'.format(
            self.train_data.y.shape))
        print('[+] Testing data dimensions')
        print('[+] Testing dataset (no. of datapoints, pixels, pixels): {}'.format(
            self.test_data.x.shape))
        print('[+] Testing dataset labels (no. of datapoints, ): {}'.format(
            self.test_data.y.shape))
        return (self.test_data.x.shape[0],
               self.train_data.x.shape[0],
               self.num_pixels,
               self.image_width,
               self.image_height)

    def normalize_x(self):
        '''
            Convert each 28x28 image to a flat continous array and
            for each RGB value [0, 255], translate it to the range
            [0, 1] to simplify neuron activation function
        '''
        new_x_train = self.train_data.x.reshape(self.num_train, self.num_pixels).astype('float32')
        new_x_test = self.test_data.x.reshape(self.num_test, self.num_pixels).astype('float32')
        new_x_train /= 255
        new_x_test /= 255
        self.train_data = Data(new_x_train, self.train_data.y)
        self.test_data = Data(new_x_test, self.test_data.y)

    def vectorize_y(self):
        '''
            Convert the y labels e.g. 7 to a vector e.g.
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        '''
        num_out_classes = 10
        y_train = keras.utils.to_categorical(self.train_data.y, num_out_classes)
        y_test = keras.utils.to_categorical(self.test_data.y, num_out_classes)
        self.train_data = Data(self.train_data.x, y_train)
        self.test_data = Data(self.test_data.x, y_test)

    def get_data(self):
        self.normalize_x()
        self.vectorize_y()
        return self.train_data, self.test_data
