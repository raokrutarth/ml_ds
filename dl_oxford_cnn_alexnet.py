

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard

import tflearn.datasets.oxflower17 as oxflower17



class OxfordFlowersAlexNet:
    '''
        Convolutional neural network in Keras to classify images of flowers

        AlexNet
        https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks

        Build a known neural network (AlexNet) to classify images of different flowers
        e.g. identify tulips, sunflowers etc. The dataset contains images of flowers
        belonging to 17 different categories. The images were acquired by searching the
        web and taking pictures. There are 80 images for each category.

        Note
            - When transitioning from a convolutional layer to a dense layer, a
                flattening layer is needed in between.
            - Using 2D strides in the max pooling layers
            - Tensorbord loss, accuracy and network visualization
    '''
    def __init__(self):
        np.random.seed(42)
        self.get_data()
        self.model = self.make_model()

    def get_data(self):
        '''
            Each image is represented as a 224x224 pixel RGB colored image. Therefore,
            each image is [3][224][224]float matrix. The dataset is already
            pre-processed so all values are (0, 1).

            Each label is a binary [17]float where each image is assigned to one of the 17
            classes of flowers.
        '''
        self.data_x, self.data_y = oxflower17.load_data(dirname="data/17flowers", one_hot=True)
        self.image_width, self.image_height, self.color_depth = \
         self.data_x.shape[1], self.data_x.shape[2], self.data_x.shape[3]
        self.n_classes = self.data_y.shape[1]
        print('[+] Loaded dataset: ')
        print('[+] Data point dimensions: ', self.data_x.shape)
        print('[+] Label dimensions: ', self.data_y.shape)


    def make_model(self):
        '''
            Setup layers and optimizer
        '''
        model = Sequential()
        l1_neurons = 96
        l4_neurons = 256
        l7_neurons = 256
        l8_neurons = 384
        l9_neurons = 384
        l13_neurons = 4096
        l15_neurons = 4096

        model.add(
            Conv2D(
                l1_neurons,
                kernel_size=(11, 11), # 11x11 pixel filter
                strides=(4, 4),
                activation='relu',
                input_shape=(self.image_width, self.image_height, self.color_depth),
            )
        )
        model.add(
            MaxPooling2D(
                pool_size=(3, 3),
                strides=(2, 2),
            )
        )
        model.add(BatchNormalization())
        model.add(
            Conv2D(
                l4_neurons,
                kernel_size=(5, 5),
                activation='relu',
            )
        )
        model.add(
            MaxPooling2D(
                pool_size=(3, 3),
                strides=(2, 2),
            )
        )
        model.add(BatchNormalization())
        model.add(Conv2D(l7_neurons, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(l8_neurons, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(l9_neurons, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(l13_neurons, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(l15_neurons, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(self.n_classes, activation='softmax'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'],
        )
        print(model.summary())
        return model

    def run(self):
        '''
            Can see the validation accuracy, loss, etc.
            plotted on tensorboard (after model is run) with:
                bash> tensorboard --logdir ./data/tf_logs/ --port 6006
            Open in browser
                http://127.0.0.1:6006/
        '''
        tensorboard = TensorBoard('data/tf_logs/alexnet')
        self.model.fit(
            self.data_x, self.data_y,
            batch_size=64,
            epochs=2,
            verbose=1,
            validation_split=0.1, # Fraction of the training data to be used as validation data.
            shuffle=True,
            callbacks=[tensorboard]
        )



def main():
    oxf_alex = OxfordFlowersAlexNet()
    oxf_alex.run()

if __name__ == '__main__':
    main()