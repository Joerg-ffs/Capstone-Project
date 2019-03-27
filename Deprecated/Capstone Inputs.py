import csv
import mss
import mss.tools
from inputs import get_gamepad

def main():
    #The screen part to capture
    #Camera Mode:6, windowed 720
    monitor = {'top': 537, 'left': 866, 'width': 191, 'height': 74}
    toggle_recording = False
    count = 0
    statusList = [0,0,0,0]
    #statusList = [0,0]
    
    def send_data(name, value):
        
        #Fetching and recording the useful inputs
        if name == "angle":
            statusList[0] = value
        elif name == "accelerator":
            statusList[1] = value
        elif name == "brake":
            statusList[2] = value
        elif name == "image":
            statusList[1] = str(output)

            #Writing the data to a csv file for training purposes
            with open(r"test_5.csv", "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(statusList)

    #Initializing mss outside of while loop improves performance   
    with mss.mss() as sct: 
        while True:
            events = get_gamepad()
            for event in events:
                if event.code == 'BTN_SELECT' and event.state == 1:
                    toggle_recording = not toggle_recording

                elif toggle_recording == True:
                    if event.code == 'ABS_Z':
                        #brake
                        send_data('brake', event.state)
                    elif event.code == 'ABS_RZ':
                        #accelerator
                        send_data('accelerator', event.state)
                    if event.code == 'ABS_X':
                        #steering angle
                        send_data('angle', event.state) 

                        #Screen Capture, and image file storage
                        count += 1
                        output = 'frames/img-{}.png'.format(count)
                        #Compressing the image
                        sct.compression_level = 2
                        # Grab the data
                        sct_img = sct.grab(monitor)
                        #Save to the picture file
                        mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
                        send_data("image", output)
                        
                elif toggle_recording == False:
                    print("Recording Paused")

if __name__ == "__main__":
    main()
