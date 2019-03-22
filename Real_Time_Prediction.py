import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mss
import mss.tools
import evdev
import cv2
from evdev import InputDevice, categorize, ecodes
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Lambda, Dropout
from keras.optimizers import Adam

IMG_WIDTH = 64
IMG_HEIGHT = 64


def main():
    # The screen part to capture
    # Camera Mode:6, windowed 720
    monitor = {'top': 537, 'left': 866, 'width': 191, 'height': 74}
    toggle_recording = False
    # Device path for steering wheel
    dev = InputDevice('/dev/input/event22')
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    count = 0
    statusList = [0, 0, 0, 0]

    # statusList = [0,0]

    def send_data(name, value):

        # Fetching and recording the useful inputs
        if name == "angle":
            statusList[0] = value
        elif name == "accelerator":
            statusList[1] = value
        elif name == "brake":
            statusList[2] = value
        elif name == "image":
            statusList[1] = str(output)

    # compile saved model
    model = Sequential()
    # Add lambda function to avoid pre-processing outside of network
    col = 64
    row = 64
    ch = 3
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(col, row, ch),
                     output_shape=(col, row, ch)))
    model.add(Conv2D(3, 5, 5, input_shape=(64, 64, 3), subsample=(2, 2), activation='relu'))
    model.add(Conv2D(24, 5, 5, input_shape=(32, 32, 3), subsample=(2, 2), activation='relu'))
    model.add(Conv2D(36, 5, 5, input_shape=(5, 22, 3), subsample=(2, 2), activation='relu'))
    model.add(Conv2D(48, 3, 3, input_shape=(3, 20, 3), activation='relu'))
    model.add(Conv2D(64, 3, 3, input_shape=(1, 18, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164, activation='linear'))
    model.add(Dropout(.5))
    model.add(Dense(100, activation='linear'))
    model.add(Dropout(.2))
    model.add(Dense(50, activation='linear'))
    model.add(Dropout(.1))
    model.add(Dense(10, activation='linear'))
    model.add(Dense(1, activation='linear'))

    model.load_weights("model_bal.h5")
    # Initializing mss outside of while loop improves performance
    with mss.mss() as sct:
        while 1:

            for event in dev.read_loop():

                if event.code == 295 and event.value != 0:
                    toggle_recording = not toggle_recording

                elif toggle_recording == True:

                    if event.code == 2:
                        # brake
                        send_data('brake', event.value)
                    elif event.code == 1:
                        # accelerator
                        send_data('accelerator', event.value)
                    if event.value != 0 and event.code == 0:
                        # steering angle
                        send_data('angle', event.value)

                        # Screen Capture, and image file storage
                        output = '/media/joerg/OS_Install/Users/Joerg/Desktop/Capstone-Project-master/predict/real_img-1.png'
                        # Compressing the image
                        sct.compression_level = 2
                        # Grab the data
                        sct_img = sct.grab(monitor)
                        # Save to the picture file
                        mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
                        send_data("image", output)

                        img_array = cv2.imread(output, 3)
                        img_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))
                        X = img_array
                        X = np.array(X).reshape(-1, IMG_WIDTH, IMG_HEIGHT,3)  # Trailing 1 represents 1 dimensions due to grayscale -> 3 for colour
                        X = X / 255.0  # Normalize image data devide by 255 pixels

                        y = model.predict(X, verbose=0)
                        os.remove(output)

                        # compare actual vs predicted angle
                        print('Actual angle: ' + str(event.value))
                        print('Predicted angle: ' + str(model.predict(np.array(X), verbose=0)))

                elif toggle_recording == False:
                    print("Recording Paused")


if __name__ == "__main__":
    main()