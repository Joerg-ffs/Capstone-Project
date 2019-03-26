import os

label = 32005
stop = 32185

while True:
    if label < stop:
        os.remove('/media/joerg/OS_Install/Users/Joerg/Desktop/Capstone-Project-master/frames/test_img-{}.png'.format(label))
        label = label + 1
    if label == stop:
        break