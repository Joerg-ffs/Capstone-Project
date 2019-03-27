"""
This program helps identify the structure of the input stream of the device that will be used to collect data
Created by Matthew Clark and Joerg Oxborough
"""

from evdev import InputDevice, categorize, ecodes
#Event number will change based on amount of connected devices
#Use the command in terminal to locate wheel: cat /proc/bus/input/devices

#Set device to event slot holding racing wheel
dev = InputDevice('/dev/input/event22')

while 1:
    
    #Event is a change in the state of one of the inputs on the wheel
    #For example, turning the steering wheel or pressing down a pedal
    for event in dev.read_loop():
        #Print useful information regarding input stream
            print(categorize(event))
            print("code: " + str(event.code) + "value: " + str(event.value))
