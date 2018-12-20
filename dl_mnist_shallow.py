
from collections import namedtuple
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

Data = namedtuple('Data', ['x', 'y'])

class ShallowNet:
    '''

        Shallow Neural Network in Keras

            A shallow neural network to classify MNIST digits. MNIST dataset is a set
            of 28x28 pixel images that contain images of handwritten numbers.

            Develop a shallow neural network to recognize the number in the image.

        Network

            In (784) -> L1 (64 N, sigmoid, dense) -> Out (10)

        Note
            - A dense layer is one where the output
                of EACH neuron goes as input to every neuron
                in the next layre.

    '''
    def __init__(self):
        np.random.seed(42)
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.train_data = Data(x_train, y_train)
        self.test_data = Data(x_test, y_test)
        self.num_test, self.num_train, self.num_pixels = self.get_dimensions()
        self.model = None

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
               self.train_data.x.shape[1] * self.train_data.x.shape[2])

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
        n_classes = 10
        y_train = keras.utils.to_categorical(self.train_data.y, n_classes)
        y_test = keras.utils.to_categorical(self.test_data.y, n_classes)
        self.train_data = Data(self.train_data.x, y_train)
        self.test_data = Data(self.test_data.x, y_test)

    def make_model(self):
        '''
            Define layers
        '''
        model = Sequential()
        layer_1_neurons = 64
        out_classes = 10
        learning_rate = 0.01
        model.add(
            Dense(
                layer_1_neurons,
                activation='sigmoid',
                input_shape=(self.num_pixels,)
            )
        )
        model.add(
            Dense(
                out_classes,
                activation='softmax'
            )
        )
        model.compile(
            loss='mean_squared_error',
            optimizer=SGD(lr=learning_rate),
            metrics=['accuracy']
        )
        print(model.summary())
        self.model = model

    def run(self):
        '''
            Train the model on the data and evaluate
            using test datasets
        '''
        self.model.fit(
            self.train_data.x,
            self.train_data.y,
            batch_size=15,
            epochs=2,
            verbose=1,
            validation_data=(self.test_data.x, self.test_data.y)
        )
        res = self.model.evaluate(self.test_data.x, self.test_data.y)
        print('Loss, Accuracy: ', res)


def main():
    sn = ShallowNet()
    sn.normalize_x()
    sn.vectorize_y()
    sn.make_model()
    sn.run()

if __name__ == '__main__':
    main()
