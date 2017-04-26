import pandas as pd
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.core import Dropout, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from util import INPUT_SHAPE, generator

np.random.seed(0)
dirimg = 'data'

def load_data():
    """
    Load training data
    """
    lines = []

    with open('./' + dirimg + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    return lines

def build_model():
    """
    Simplified NVIDIA model 
    """
    model = Sequential()

    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))

    model.add(Flatten())
    model.add(Dense(50))
    model.add(Dropout(.5))
    model.add(Dense(10))
    model.add(Dropout(.5))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model


def train_model(model, lines):
    """
    Train the model
    """
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    train_generator = generator(train_samples, batch_size=18)
    validation_generator = generator(validation_samples, batch_size=18)
    checkpoint = ModelCheckpoint('model.h5',
                                     monitor='val_loss',
                                     verbose=0,
                                     save_best_only=True,
                                     mode='auto')

    model.fit_generator(train_generator, samples_per_epoch=len(train_samples) * 3,
                        validation_data=validation_generator,
                        nb_val_samples=len(validation_samples) * 3, nb_epoch=16,
                        callbacks=[checkpoint])


def main():

    lines = load_data()

    model = build_model()

    train_model(model, lines)


if __name__ == '__main__':
    main()
