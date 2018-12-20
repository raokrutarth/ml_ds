

from collections import namedtuple
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten, Conv2D, MaxPooling2D

# local pre-processor for commonly used dataset
from mnist_preprocess import MINSTData

class MNISTLeNet5:
    '''

        Convolutional Neural Network in Keras

            A convolutional neural network to classify MNIST digits. MNIST dataset is a set
            of 28x28 pixel images that contain images of handwritten numbers.

            Develop a neural network to recognize the number in the image.

        Network
            LeNet 5 from http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

            In (784) -> ... -> Out (10)

        Note
            - 2D Convolutional layer
            - Max pooling layer: for a AxA grid of outputs, it takes the single most
                prominant/(largest?) output.
            - Flattening converts the 2D output of a convolutional layer to a 1D
                output.
            - filters = kernels = neurons
    '''
    def __init__(self):
        np.random.seed(42)
        self.n_classes = 10
        self.load_data()
        self.model = self.make_model()

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.image_height = x_train.shape[1]
        self.image_width = x_train.shape[2]
        self.color_depth = 1 # 3 for color images
        self.num_train, self.num_test = x_train.shape[0], x_test.shape[0]
        # convert images to 2D black/white image representation
        x_train = x_train.reshape(
            self.num_train, self.image_height, self.image_width, self.color_depth
        ).astype('float32')
        x_test = x_test.reshape(
            self.num_test, self.image_height, self.image_width, self.color_depth
        ).astype('float32')
        x_train /= 255
        x_test /= 255
        # convert labels to vectors
        y_train = keras.utils.to_categorical(y_train, self.n_classes)
        y_test = keras.utils.to_categorical(y_test, self.n_classes)
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def make_model(self):
        '''
            Define layers
        '''
        l1_neurons = 32
        l2_neurons = 64
        l6_neurons = 128
        model = Sequential()
        model.add(
            Conv2D(
                l1_neurons,
                kernel_size=(3, 3), # 3x3 pixel kernel
                activation='relu',
                input_shape=(
                    self.image_width,
                    self.image_height,
                    self.color_depth,
                )
            )
        )
        model.add(
            Conv2D(
                l2_neurons,
                kernel_size=(3, 3),
                activation='relu'
            )
        )
        model.add(
            MaxPooling2D(
                pool_size=(2, 2) # 1 output for every 4 pixels
            )
        )
        model.add(Dropout(0.25)) # ignore quarter of the inputs
        model.add(Flatten())
        model.add(Dense(l6_neurons, activation='relu'))
        model.add(Dropout(0.5)) # ignore half the inputs
        model.add(Dense(self.n_classes, activation='softmax'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        print(model.summary())
        return model

    def run(self):
        '''
            Train the model on the data and evaluate
            using test datasets
        '''
        self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=128,
            epochs=2,
            verbose=1,
            validation_data=(self.x_test, self.y_test)
        )
        res = self.model.evaluate(self.x_test, self.y_test)
        print('Test Loss, Accuracy: ', res)


def main():
    sn = MNISTLeNet5()
    sn.run()

if __name__ == '__main__':
    main()

