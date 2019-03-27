import evdev
from evdev import InputDevice, categorize, ecodes

dev = InputDevice('/dev/input/event22')
devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
while 1:
    for event in dev.read_loop():
        if event.code ==1 and event.value != 0:
            print(categorize(event))
            print(event.value)
            print("code: " + str(event.code))
