"""
This program collects the input values needed for Tier 2 of the final prototype.
Created by Matthew Clark and Joerg Oxborough
"""

import csv
import numpy as np
import mss
import mss.tools
import cv2
from evdev import InputDevice
from keras.models import load_model
from keras import backend as K

#This function is needed in order to load saved Tier 1 model
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def main():
    
    #Device path for steering  
    dev = InputDevice('/dev/input/event22')         #Event number will change based on amount of connected devices cat /proc/bus/input/devices
    
    #Area for screen capture
    monitor = {'top': 437, 'left': 766, 'width': 400, 'height': 300}
    
    #Saved Tier 1 model
    model = load_model('Tier_1_Model.h5', custom_objects={'coeff_determination': coeff_determination})   
    
    #List for collected inputs
    statusList = [0, 0, 0, 0, 0]
    prev_angles = []    
    toggle_recording = False
    
    #This function collects the input data into a list and writes it to a csv
    def send_data(name, value):
        
        # Fetching and recording the useful inputs
        if name == "angle":
            statusList[0] = value
        elif name == 'prediction':
            statusList[1] = value
        elif name == "accelerator":
            statusList[2] = value
        elif name == "brake":
            statusList[3] = value
            
        # Writing the data to a csv file for training purposes
        with open(r"non-distracted_1.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(statusList)
            
            
    with mss.mss() as sct:
        while True:
            
            #Event is a change in the state of one of the inputs on the wheel
            #For example, turning the steering wheel or pressing down a pedal
            for event in dev.read_loop():
                
                #Button status to toggle recording on and off   
                if event.code == 295  and event.value != 0:
                    toggle_recording = not toggle_recording
                
                elif toggle_recording == True:
                    
                    # brake
                    if event.code == 2:
                        send_data('brake', event.value)
                        
                    # accelerator    
                    elif event.code == 1:                        
                        send_data('accelerator', event.value)
                        
                    # steering angle
                    if event.value != 0 and event.code == 0:
                        send_data('angle', event.value)

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
                        img_array = cv2.resize(img_array, (200, 66))
                        X = np.array(img_array).reshape(-1, 66, 200, 3)
                        
                        
                        #use Tier One model to predict angle for current image
                        pred_angle = model.predict(X, batch_size=1, verbose=0)[0][0]
                        
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
                        
                        #send smoothed angle to csv
                        send_data('prediction', smooth_angle)

                elif toggle_recording == False:
                    print("Recording Paused")


if __name__ == "__main__":
    main()
