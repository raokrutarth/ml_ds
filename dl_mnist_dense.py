
from collections import namedtuple
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# local precrocessor for commonly used dataset
from mnist_preprocess import MINSTData

class DenseNet:
    '''

        Dense Neural Network in Keras

            A dense neural network to classify MNIST digits. MNIST dataset is a set
            of 28x28 pixel images that contain images of handwritten numbers.

            Develop a neural network to recognize the number in the image.

        Network

            In (784) -> L1 (relu, 64n) -> L2 (relu, 64n) -> Out (10)

        Note
            Similar to shallow network in dl_mnist_shallow.py with:
            - One more dense layre
            - Using cross entropy cost instead of mean squared error
            - ReLU activation on both layers

        With one extra dense layer and a "better" activation function,
        2 epoch accuracy jumps drastically (94%).

    '''
    def __init__(self):
        np.random.seed(42)
        mnist_preprocess = MINSTData()
        self.num_test, self.num_train, self.num_pixels, _, _ = mnist_preprocess.get_dimensions()
        self.train_data, self.test_data = mnist_preprocess.get_data()
        self.model = None

    def make_model(self):
        '''
            Make model with 2 dense layers. The output
            of EACH neuron goes as input to every neuron
            in the next layre.

            784 inputs -> layer 1 (relu, 64 neurons)
            layer 1 -> layer 2 (relu, 64 neurons)
            layer 2 -> output layre (softmax, 10 classes/buckets)
        '''
        model = Sequential()
        layer_1_neurons = 64
        layer_2_neurons = 64
        out_classes = 10
        learning_rate = 0.01
        model.add(
            Dense(
                layer_1_neurons,
                activation='relu',
                input_shape=(self.num_pixels,)
            )
        )
        model.add(
            Dense(
                layer_2_neurons,
                activation='relu',
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
            loss='categorical_crossentropy',
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
    sn = DenseNet()
    sn.make_model()
    sn.run()

if __name__ == '__main__':
    main()
