import cv2
import numpy as np
import keras
from keras.layers import Dense, Flatten, Conv2D, Lambda
from keras.layers import Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import pandas as pd
import random
from keras import backend as K
import math


data = pd.read_csv("comma_random.csv")
data_img = np.array(data['Frame'])
data_img = data_img.reshape(-1, 1)
data_angle = np.array(data['Angle'])
data_angle = data_angle.reshape(-1, 1)
print(data_img.shape)
data_img = data_img[0:50000]
data_angle = data_angle[0:50000]
print(data_img.shape)
batch_size = 32
num_samples = data_angle.shape[0]
print(num_samples)
STEPS_PER_EPOCH = math.ceil(num_samples / batch_size)


def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


class SensitivitySpecificityCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 1:
            x_test = self.validation_data[0]
            y_test = self.validation_data[1]
            # x_test, y_test = self.validation_data
            predictions = self.model.predict(x_test)
            print(coeff_determination(y_test, predictions))


def comma_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(120, 240, 3)))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=Adam(lr=0.00001), metrics=[coeff_determination])
    filepath = "Capstone.h5"
    checkpoint1 = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
    callbacks_list = [checkpoint1]
    return model, callbacks_list


def generator(batch_size):
    # Create empty arrays to contain batch of features and labels#
    test_size = 2 * round((batch_size * 4 / 5) / 2)

    batch_features = np.zeros((test_size * 2, 120, 240, 3))
    batch_labels = np.zeros((test_size * 2, 1))

    while True:
        for i in range(test_size):
            # choose random index in features
            index = random.randrange(1, num_samples)
            batch_features[2 * i - 1] = cv2.imread('comma1/' + data_img[index][0] + '.jpg', cv2.IMREAD_COLOR)
            batch_labels[2 * i - 1] = data_angle[index][0]
            batch_features[2 * i] = cv2.flip(cv2.imread('comma1/' + data_img[index][0] + '.jpg', cv2.IMREAD_COLOR), 1)
            batch_labels[2 * i] = data_angle[index][0] * -1

        yield batch_features, batch_labels

def val_generator(batch_size):
    val_size = 2 * round((batch_size * 1 / 5) / 2)

    batch_features = np.zeros((val_size * 2, 120, 240, 3))
    batch_labels = np.zeros((val_size * 2, 1))

    while True:
        for i in range(val_size):
            # choose random index in features
            index = random.randrange(1, num_samples)
            batch_features[2 * i - 1] = cv2.imread('comma1/' + data_img[index][0] + '.jpg', cv2.IMREAD_COLOR)
            batch_labels[2 * i - 1] = data_angle[index][0]
            batch_features[2 * i] = cv2.flip(cv2.imread('comma1/' + data_img[index][0] + '.jpg', cv2.IMREAD_COLOR), 1)
            batch_labels[2 * i] = data_angle[index][0] * -1

        yield batch_features, batch_labels


model, callbacks_list = comma_model()
# model = load_model('Capstone.h5', custom_objects={'coeff_determination': coeff_determination})
model.fit_generator(generator(batch_size), validation_data=val_generator(batch_size), steps_per_epoch=STEPS_PER_EPOCH,
                    validation_steps=STEPS_PER_EPOCH, epochs=200, callbacks=callbacks_list, max_queue_size=200)
# model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=200,callbacks=callbacks_list)