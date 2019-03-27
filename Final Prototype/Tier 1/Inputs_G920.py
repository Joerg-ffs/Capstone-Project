"""
This program gathers steering wheel and image data for input into the neural net.
Created by Joerg Oxborough and Matthew Clark
"""

import csv
import mss
import mss.tools
import time
from evdev import InputDevice

def main():
    # The screen part to capture
    # Camera Mode:1, windowed 720
    monitor = {'top': 437, 'left': 766, 'width': 400, 'height': 300}
    
    #Device path for steering wheel
    dev = InputDevice('/dev/input/event22')     
    
    #Boolean for recording status    
    toggle_recording = False    
    
    #number of frames and angles recorded                        
    count = 0
    
    #list for storing angle and image values
    statusList = [0, 0]
    
    def send_data(name, value):

        # Fetching and recording the useful inputs
        
        if name == "angle":
            #Records the angle to the first position of the list
            statusList[0] = value
        
        elif name == "image":
            #records the path of the image in the second position of the list
            statusList[1] = str(value)

            # Writing the list to a csv file for input into model
            with open(r"tier_1_inputs.csv", "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(statusList)

    # Initializing mss, screen capture library
    with mss.mss() as sct:
        while True:
            #Event is a change in the state of one of the inputs on the wheel
            #For example, turning the steering wheel or pressing down a pedal
            
            for event in dev.read_loop():
                
                #Button status to toggle recording on and off
                if event.code == 295  and event.value != 0:
                    toggle_recording = not toggle_recording
            
                elif toggle_recording == True:
                    
                    #Grab steering wheel values
                    #Steering wheel outputs some 0 values erraticly so they are filtered out
                    if event.value != 0 and event.code == 0:
                        
                        # steering angle
                        send_data('angle', event.value)

                        # Screen Capture, and image file storage
                        count += 1
                        output = '/media/joerg/OS_Install/Users/Joerg/Desktop/Capstone-Project-master/frames/test_img-{}.png'.format(count)
                        # Compressing the image
                        sct.compression_level = 2
                        # Grab the data
                        sct_img = sct.grab(monitor)
                        # Save to the picture file
                        mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
                        #send path to csv file
                        send_data("image", output)
                        #slows down recording to around 15 frames a second
                        time.sleep(0.065)
                        
                elif toggle_recording == False:
                    print("Recording Paused")


if __name__ == "__main__":
    main()
