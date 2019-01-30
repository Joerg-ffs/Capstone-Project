import pygame
import math
import os
from http.cookiejar import CookieJar
import numpy
import pandas as pd

# make sure pygame doesn't try to open an output window
os.environ["SDL_VIDEODRIVER"] = "dummy"

DEBUG = False
HOST = "localhost"
PORT = "50000"
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

gear_lever_position = 0
parking_brake_status = False
headlamp_status = False
high_beam_status = False
windshield_wiper_status = False
axis_mode = 1
ButtonList = []
statusList = [0,0,0,0]
def send_data(name, value):
    #Fetching and recording the useful inputs
    if name == "angle":
        statusList[0] = value
    elif name == "accelerator":
        statusList[1] = value
    elif name == "brake":
        statusList[2] = value
    elif name == "distraction":
        statusList[3] = value
    ButtonList.append(statusList)
    
    #Writing the data to a text file for training purposes
    with open("Distracted-Joerg-Highway-1.txt", "a") as myfile:
        myfile.write(str(statusList))
  

def pedal_value(value):
  '''Steering Wheel returns pedal reading as value 
  between -1 (fully pressed) and 1 (not pressed)
  normalizing to value between 0 and 100%'''
  return (1 - value) * 50
toggle = 0
pygame.init()                        

try: 

  wheel = None
  for j in range(0,pygame.joystick.get_count()):
    if pygame.joystick.Joystick(j).get_name() == WHEEL:
      wheel = pygame.joystick.Joystick(j)
      wheel.init()
      print("Found", wheel.get_name())

  if not wheel:
    print("No G27 steering wheel found")
    exit(-1)

  while True:
    for event in pygame.event.get(pygame.QUIT):
      exit(0)
    for event in pygame.event.get():
      if event.type == pygame.KEYDOWN:
          if event.key == pygame.K_KP_ENTER:
              toggle = 0
          elif event.key == pygame.K_KP_PLUS:
              toggle = 1
          send_data("distraction", toggle)      
      if toggle == 0:
          print('0')
      elif toggle == 1:
          print('1')
    for event in pygame.event.get(pygame.JOYAXISMOTION):
        if DEBUG:
            print("Motion on axis: ", event.axis)
        if event.axis == 0:
            send_data("angle", event.value * 600)   
        elif event.axis == 2:
            send_data("accelerator", pedal_value(event.value))
        elif event.axis == 3:
            send_data("brake", pedal_value(event.value))
    for event in pygame.event.get(pygame.JOYBUTTONUP):
      if DEBUG:
        print("Released button is", event.button)
      if (event.button >= 12 and event.button <= 17) or event.button == 22:
        gear = 0
        send_data("gear_lever_position", gear_lever_positions[gear])
    for event in pygame.event.get(pygame.JOYBUTTONDOWN):
      if DEBUG:
        print("Pressed button is", event.button)
      if event.button == 0: 
        print("pressed button 0 - bye...")
        print(ButtonList)
      elif event.button == 11:
        send_data("ignition_status","start")
      elif event.button >= 12 and event.button <= 17:
        gear = event.button - 11
        send_data("gear_lever_position", gear_lever_positions[gear])
      elif event.button == 22:
        gear = -1
        send_data("gear_lever_position", gear_lever_positions[gear])
      elif event.button in status_buttons:
        vars()[status_buttons[event.button]] = not vars()[status_buttons[event.button]]
        send_data(status_buttons[event.button],str(vars()[status_buttons[event.button]]).lower())

except Exception as e:
  print(e)