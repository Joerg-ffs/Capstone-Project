import csv
import mss
import mss.tools
import evdev
from evdev import InputDevice, categorize, ecodes

def main():
    # The screen part to capture
    # Camera Mode:6, windowed 720
    monitor = {'top': 537, 'left': 866, 'width': 191, 'height': 74}
    toggle_recording = False
    #Device path for steering wheel
    dev = InputDevice('/dev/input/event17')
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

            # Writing the data to a csv file for training purposes
            with open(r"test_5.csv", "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(statusList)

    # Initializing mss outside of while loop improves performance
    with mss.mss() as sct:
        while True:
            for event in dev.read_loop():

                if event.code == 295  and event.value != 0:
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
                        count += 1
                        output = '/media/joerg/OS_Install/Users/Joerg/Desktop/Capstone-Project-master/frames/test_img-{}.png'.format(count)
                        # Compressing the image
                        sct.compression_level = 2
                        # Grab the data
                        sct_img = sct.grab(monitor)
                        # Save to the picture file
                        mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
                        send_data("image", output)

                elif toggle_recording == False:
                    print("Recording Paused")


if __name__ == "__main__":
    main()