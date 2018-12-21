
from collections import namedtuple
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras import regularizers

# local pre-processor for commonly used dataset
from mnist_preprocess import MINSTData

class MNISTDeepNet:
    '''

        Deep Neural Network in Keras

            A dense neural network to classify MNIST digits. MNIST dataset is a set
            of 28x28 pixel images that contain images of handwritten numbers.

            Develop a basic deep neural network to recognize the number in the image.

        Network

            In (784) -> L1 (64n, dense, relu, weight regularization & normal dist. initilization)
            L1 -> L2 (Batch normalize)
            L2 -> L3 (Dropout, 0.5)
            L3 -> L4 (64n, dense, relu)
            L4 -> L5 (Batch normalize)
            L5 -> L6 (Dropout, 0.5)
            L6 -> L7 (128n, dense, tanh)
            L7 -> L8 (Dropout, 0.5)
            L8 -> Out (10)

        Note
            Similar to shallow and deep network in dl_mnist_shallow.py and
            dl_mnist_deep.py with:
            - Adam optimization to make sure learning rate/step is adjusted as
                gradient descent proceeds.
            - Batch Normalization:
                - Makes sure the distribution of inputs to layers does
                    not shift layer-by-layer. This tries to prevent covariate shift. i.e.
                    neurons getting stuck at high or low output levels (aka saturation).
                - Mimimizes the effects of the random weight initilizations.
                - On some ocassions, avoids having to use dropout layers.
            - Dropout layers randomly drop some parts of the input to the neurons to avoid
                overfitting.
            - Regularization adds a penalty for using high weights.
                - regularization on weights is called kernel regularization
            - Initilization of the random weights for each layer can effect on
                the time to learn so we can set that as well. Default is uniform
                distribution.
    '''
    def __init__(self):
        np.random.seed(42)
        mnist_preprocess = MINSTData()
        self.num_test, self.num_train, self.num_pixels, _, _ = mnist_preprocess.get_dimensions()
        self.train_data, self.test_data = mnist_preprocess.get_data()
        self.model = None

    def make_model(self):
        '''
            Define layers
        '''
        layer_1_neurons = 64
        layer_2_neurons = 64
        layer_3_neurons = 128
        out_classes = 10
        dropout_ratio = 0.5
        model = Sequential()
        model.add(
            Dense(
                layer_1_neurons,
                activation='relu',
                input_shape=(self.num_pixels,),
                kernel_regularizer=regularizers.l2(0.01), # example of setting weight regularizer
                kernel_initializer='glorot_normal', # example of setting weight initilizer
            )
        )
        model.add(BatchNormalization())
        model.add(Dropout(dropout_ratio))
        model.add(Dense(layer_1_neurons, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_ratio))
        model.add(Dense(layer_2_neurons, activation='tanh'))
        model.add(BatchNormalization())
        model.add(Dense(layer_3_neurons, activation='relu'))
        model.add(Dropout(dropout_ratio))
        model.add(
            Dense(
                out_classes,
                activation='softmax'
            )
        )
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
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
            batch_size=128,
            epochs=2,
            verbose=1,
            validation_data=(self.test_data.x, self.test_data.y)
        )
        res = self.model.evaluate(self.test_data.x, self.test_data.y)
        print('Test Loss, Accuracy: ', res)


def main():
    sn = MNISTDeepNet()
    sn.make_model()
    sn.run()

if __name__ == '__main__':
    main()
