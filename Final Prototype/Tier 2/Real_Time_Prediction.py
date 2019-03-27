"""
This program attempts to predict distraction level in real time using screen captures and inputs from a racing wheel
Created by Matthew Clark and Joerg Oxborough
"""

import numpy as np
import pickle
import mss
import mss.tools
import cv2
from evdev import InputDevice
from keras.models import load_model
from keras import backend as K

DISTRACTION_THRESHOLD = 75

#This function is needed in order to load saved Tier 1 model
def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))

def main():
    
    #Loading the previously trained Tier 1 model
    model = load_model('Tier_1_model.h5', custom_objects={'coeff_determination': coeff_determination})
    
    # Loading the previously trained Tier 2 model
    ERF = pickle.load(open('Tier_2_model.pkl', 'rb'))
    
    # The screen part to capture
    # Camera Mode:1, windowed 720
    monitor = {'top': 437, 'left': 766, 'width': 400, 'height': 300}
    
    # Device path for steering wheel
    dev = InputDevice('/dev/input/event22')
    
    #List for collected inputs
    statusList = [0, 0, 0, 0]
    
    prev_angles = []
    toggle_recording = False
    inputs_array = []
    prediction = 0
    j =0
    sum_pred = 0

    #This function collects the input data into a list and returns the list
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
            
            #Event is a change in the state of one of the inputs on the wheel
            #For example, turning the steering wheel or pressing down a pedal
            for event in dev.read_loop():
                
                #Button status to toggle recording on and off  
                if event.code == 295 and event.value != 0:
                    toggle_recording = not toggle_recording

                elif toggle_recording == True:
                    
                    # brake
                    if event.code == 2:
                        send_data('brake', event.value)
                        
                    # accelerator    
                    elif event.code == 1:                        
                        send_data('accelerator', event.value)
                        
                    # steering angle
                    if event.code == 0 and event.value != 0:
                        
                        # Screen Capture, and image file storage
                        output = '/media/joerg/A6CA2428CA23F36B/predict/test.png'
                        # compressing the image
                        sct.compression_level = 2
                        # grab the data
                        sct_img = sct.grab(monitor)
                        # save to the picture file
                        mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
                        
                         #Converts image to an array, crops, and resizes image to 200x66 from 400x300
                        img_array = cv2.imread(output, 3)
                        img_array = img_array[80:212, 0:400]
                        img_array = cv2.resize(img_array, (66, 200))
                        X = np.array(img_array).reshape(-1, 66, 200, 3)
                        
                        #use Tier 1 model to predict angle for current image
                        pred_angle = model.predict(X, batch_size=1, verbose=0)[0][1]
                                                  
                        #normalize current angle
                        angle = (event.value - 32752) / 32752 * 2

                        #smoothing function for predicted angle, rolling array of past 20 angles
                        if len(prev_angles) < 20:
                            prev_angles.append(pred_angle)
                            prev_angles.append(angle)

                        else:
                            prev_angles.pop(0)
                            prev_angles.pop(1)
                            prev_angles.append(pred_angle)
                            prev_angles.append(angle)
                            
                        smooth_angle = sum(prev_angles) / len(prev_angles)
                        
                        #get list of all inputs
                        inputs = send_data('angles', [angle, smooth_angle])
                        inputs = np.array(inputs)
                        inputs_array = np.array(inputs_array)
                        
                        #The Tier 2 model was trained in sets of 100 inputs, each input list is 4 long so 4x100=400
                        #This collects the inputs into a rolling array of 100 sets
                        if np.size(inputs_array, 0) < 400:
                            inputs_array = np.append(inputs_array, inputs)
                        else:
                            inputs_array = np.array(inputs_array)
                            inputs_array = np.delete(inputs_array, [0, 1, 2, 3])
                            inputs_array = np.append(inputs_array, inputs)
                            inputs_array = inputs_array.reshape(1, -1)
                            
                            #predicts the probability of distractedness based on the 100 sets of inputs
                            prediction =  ERF.predict_proba(np.array(inputs_array))[0][0][1]
                            sum_pred += prediction
                            inputs_array = np.delete(inputs_array, [0, 1, 2, 3])
                            
                        #Takes the average of the last 50 predictions, if greater than threshold print distracted
                        if j % 50 == 0:
                            if sum_pred*2 > DISTRACTION_THRESHOLD:
                                print('DISTRACTED')
                            sum_pred = 0
                        j += 0.5
                        
                elif toggle_recording == False:
                    print("Recording Paused")


if __name__ == "__main__":
    main()
