import pygame
import os
from time import time
import mss
import mss.tools

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

#Initalize pygame
pygame.init()

# Set the width and height of the screen [width,height]
size = [1, 1]
screen = pygame.display.set_mode(size)

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

# Initialize the joysticks
pygame.joystick.init()


DEBUG = False
WHEEL = "Logitech G920 Driving Force Racing Wheel USB"
gear_lever_positions = {
    -1: "reverse",
    0: "neutral",
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth"
}

status_buttons = {
    10: "parking_brake_status",
    1: "headlamp_status",
    3: "high_beam_status",
    2: "windshield_wiper_status"
}

start = time()
count = 0

# The screen part to capture
monitor = {'top': 144, 'left': 80, 'width': 1397, 'height': 782}

gear_lever_position = 0
parking_brake_status = False
headlamp_status = False
high_beam_status = False
windshield_wiper_status = False
axis_mode = 1
ButtonList = []
statusList = [0,0,0,0,0]

def send_data(name, value):
    #Fetching and recording the useful inputs
    if name == "angle":
        statusList[0] = value
    elif name == "accelerator":
        statusList[1] = value
    elif name == "brake":
        statusList[2] = value
    elif name == "distraction":
        statusList[3] = int(value)
    elif name == "image":
        statusList[4] = str(output)
    print(statusList)
    ButtonList.append(statusList)
  
    #Writing the data to a text file for training purposes
    with open("Distracted-Joerg-Highway-1.txt", "a") as myfile:
        myfile.write(str(statusList))


def pedal_value(value):
  '''Steering Wheel returns pedal reading as value
  between -1 (fully pressed) and 1 (not pressed)
  normalizing to value between 0 and 100%'''
  return (1 - value) * 50

toggle_distraction = False
toggle_recording = True
pygame.init()

#wheel = None
#for j in range(0,pygame.joystick.get_count()):
    #if pygame.joystick.Joystick(j).get_name() == WHEEL:
        #wheel = pygame.joystick.Joystick(j)
        #wheel.init()
        #print("Found", wheel.get_name())

#if not wheel:
    #print("No steering wheel found")
    #toggle_recording = False

while toggle_recording:

# EVENT PROCESSING STEP
    for event in pygame.event.get():  # User did something
        if event.type == pygame.QUIT:  # If user clicked close
            toggle_recording = False
        # Possible joystick actions: JOYAXISMOTION JOYBALLMOTION JOYBUTTONDOWN JOYBUTTONUP JOYHATMOTION
        if event.type == pygame.JOYAXISMOTION:
            if event.axis == 0:
                send_data("angle", round(event.value, 3))
            elif event.axis == 2:
                if event.value < 0:
                    send_data("accelerator", round(event.value, 3))
                    send_data("brake", 0)
                elif event.value > 0:
                    send_data("brake", round(event.value, 3))
                    send_data("accelerator", 0)

    # DRAWING STEP
    screen.fill(WHITE)

    # Get count of joysticks
    joystick_count = pygame.joystick.get_count()

    # For each joystick:
    for i in range(joystick_count):
        joystick = pygame.joystick.Joystick(i)
        joystick.init()
        #print('Found', joystick.get_name())

        # Get the name from the OS for the controller/joystick
        name = joystick.get_name()

        # Usually axis run in pairs, up/down for one, and left/right for
        axes = joystick.get_numaxes()
        for i in range(axes):
            axis = (joystick.get_axis(i) + 1.0) / 2.0

    pygame.display.flip()

    # Limit to 30 frames per second
    clock.tick(30)

    with mss.mss() as sct:   
        output = 'frames/sct-{}.png'.format(count)
        count += 1

        # Grab the data
        sct_img = sct.grab(monitor)

        # Save to the picture file
        mss.tools.to_png(sct_img.rgb, sct_img.size, output)
        send_data("image", output)

for event in pygame.event.get(pygame.QUIT):
    exit(0)

    #Keyboard inputs
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_KP_ENTER:
            toggle_distraction = not toggle_distraction
        if event.key == pygame.K_SPACE:
            toggle_recording = not toggle_recording
        send_data("distraction", toggle_distraction)

    #steering wheel + pedal inputs
    for event in pygame.event.get(pygame.JOYAXISMOTION):
        if event.axis == 0:
            send_data("angle", event.value * 600)
        elif event.axis == 2:
            send_data("accelerator", pedal_value(event.value))
        elif event.axis == 3:
            send_data("brake", pedal_value(event.value))