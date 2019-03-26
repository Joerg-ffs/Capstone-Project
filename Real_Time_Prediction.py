import os
import csv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mss
import mss.tools
import evdev
import cv2
from evdev import InputDevice
from keras.models import load_model
from keras import backend as K

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def main():
    # The screen part to capture
    # Camera Mode:6, windowed 720
    monitor = {'top': 437, 'left': 766, 'width': 400, 'height': 300}
    toggle_recording = False
    # Device path for steering wheel
    dev = InputDevice('/dev/input/event22')
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    prev_angles = []
    model = load_model('Capstone2.h5', custom_objects={'coeff_determination': coeff_determination})
    # Initializing mss outside of while loop improves performance
    with mss.mss() as sct:
        while 1:
            for event in dev.read_loop():

                if event.code == 295 and event.value != 0:
                    toggle_recording = not toggle_recording

                elif toggle_recording == True:
                        if event.code == 0 and event.value != 0:
                            # steering angle
                            #send_data('angle', event.value)

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
                            angle = (event.value-32752)/32752 * 2
                            if len(prev_angles) < 20:
                                prev_angles.append(pred_angle)
                                prev_angles.append(angle)

                            else:
                                prev_angles.pop(0)
                                prev_angles.pop(1)
                                prev_angles.append(pred_angle)
                                prev_angles.append(angle)
                            smooth_angle = sum(prev_angles)/len(prev_angles)
                            # compare actual vs predicted angle
                            #print('Actual angle: ' + str(angle))
                            #print('Predicted angle: ' + str(smooth_angle))
                            print('Difference: ' + str(abs(smooth_angle - angle)))
                            saveData = [angle, smooth_angle]
                            
                            # Writing the data to a csv file for training purposes
                            time.sleep(0.025)
                            with open(r"prediction5.csv", "a", newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow(saveData)

                elif toggle_recording == False:
                    print("Recording Paused")


if __name__ == "__main__":
    main()