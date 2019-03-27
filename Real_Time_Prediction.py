import os
import csv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import mss
import mss.tools
import evdev
import cv2
from evdev import InputDevice
from keras.models import load_model
from keras import backend as K

inputs_array = []

def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def main():
    # Loading the previously trained model
    ERF = pickle.load(open('ERFtest3.pkl', 'rb'))
    inputs_array = []
    # The screen part to capture
    # Camera Mode:6, windowed 720
    monitor = {'top': 437, 'left': 766, 'width': 400, 'height': 300}
    toggle_recording = False
    # Device path for steering wheel
    dev = InputDevice('/dev/input/event22')
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    statusList = [0, 0, 0, 0]
    prev_angles = []
    prev_pred = []
    prediction = 0
    model = load_model('Capstone2.h5', custom_objects={'coeff_determination': coeff_determination})
    j =0
    sum_pred = 0
    # Initializing mss outside of while loop improves performance

    def send_data(name, value):
        # Fetching and recording the useful inputs
        if name == "angles":
            statusList[0] = value[0]
            statusList[1] = value[1]
        elif name == "accelerator":
            statusList[2] = value
        elif name == "brake":
            statusList[3] = value
        return statusList

    with mss.mss() as sct:
        while 1:
            for event in dev.read_loop():

                if event.code == 295 and event.value != 0:
                    toggle_recording = not toggle_recording

                elif toggle_recording == True:
                    if event.code == 0 and event.value != 0:
                        # Screen Capture, and image file storage
                        output = '/media/joerg/A6CA2428CA23F36B/predict/test.png'
                        # compressing the image
                        sct.compression_level = 2
                        # grab the data
                        sct_img = sct.grab(monitor)
                        # save to the picture file
                        mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)

                        img_array = cv2.imread(output, 3)
                        img_array = img_array[80:212, 0:400]
                        img_array = cv2.resize(img_array, (66, 200))
                        X = np.array(img_array).reshape(-1, 66, 200, 3)
                        pred_angle = model.predict(X, batch_size=1, verbose=0)[0][0]
                        angle = (event.value - 32752) / 32752 * 2

                        if len(prev_angles) < 20:
                            prev_angles.append(pred_angle)
                            prev_angles.append(angle)

                        else:
                            prev_angles.pop(0)
                            prev_angles.pop(1)
                            prev_angles.append(pred_angle)
                            prev_angles.append(angle)
                        smooth_angle = sum(prev_angles) / len(prev_angles)
                        # compare actual vs predicted angle
                        # print('Actual angle: ' + str(angle))
                        # print('Predicted angle: ' + str(smooth_angle))

                        inputs = send_data('angles', [angle, smooth_angle])
                        inputs = np.array(inputs)
                        inputs_array = np.array(inputs_array)
                        #print(inputs.shape)
                        if np.size(inputs_array, 0) < 400:
                            #print(inputs_array.shape)
                            inputs_array = np.append(inputs_array, inputs)
                        else:

                            inputs_array = np.array(inputs_array)
                            #print(inputs_array[0:4])
                            inputs_array = np.delete(inputs_array, [0, 1, 2, 3])

                            inputs_array = np.append(inputs_array, inputs)


                            inputs_array = inputs_array.reshape(1, -1)

                            prediction =  ERF.predict_proba(np.array(inputs_array))[0][0][1]
                            sum_pred += prediction
                            #print('J: ' + str(j) + ' sum: ' + str(sum_pred))
                            inputs_array = np.delete(inputs_array, [0, 1, 2, 3])
                            #print(inputs_array.shape)
                        # Writing the data to a csv file for training purposes
                        if len(prev_pred) < 50:
                            prev_pred.append(prediction)
                        else:
                            prev_pred.pop(0)
                            prev_pred.append(prediction)

                        smooth_pred = sum(prev_pred) / len(prev_pred)
                        if j % 50 == 0:
                            print(sum_pred*2)
                            if sum_pred*2 > 75:
                                print('DISTRACTED')
                            sum_pred = 0

                        #if smooth_pred > 0.8:
                            #print('Distracted')
                        j += 0.5
                elif toggle_recording == False:
                    print("Recording Paused")


if __name__ == "__main__":
    main()