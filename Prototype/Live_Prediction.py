import winsound #Output audio on windows devices
import matplotlib.animation as animation #Create updating graphs
import inputs #Get Xbox controller input 
import re #Reformat incoming data
import matplotlib.pyplot as plt #Plot and manipulate data
import numpy as np #Mathimatical manipulation with arrays
import pandas as pd #Creating Dataframes
import pickle #Saving, and loading models
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn import model_selection #Selecting models
from sklearn import preprocessing   #Preprocessing data for models
from sklearn.metrics import classification_report #Outputing model validiaty
from sklearn.metrics import confusion_matrix #Used to pick which data is trained/tested
from sklearn.metrics import accuracy_score #Model accuracy
from sklearn.linear_model import LogisticRegression #Logistic Regression model
from sklearn.tree import DecisionTreeClassifier #Decision Tree model
from sklearn.neighbors import KNeighborsClassifier #Nearest Neighbours model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #LDA model
from sklearn.naive_bayes import GaussianNB #NB model
from sklearn.svm import SVC #Support Vector Machine (takes too much processing power to run)
from sklearn.ensemble import RandomForestClassifier #Random Forest model
from sklearn.ensemble import ExtraTreesClassifier #Extreme Random Forest model
from sklearn.ensemble import AdaBoostClassifier #ADA boost for Decision Tree

#Loading the previously trained model
ERF = pickle.load( open('ERFtest.pkl', 'rb'))
#Creating a list to count distractions
countList = []
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 200  # Set Duration To 1000 ms == 1 second

#Classifying button types
EVENT_ABB = (
    # D-PAD, aka HAT
    ('Absolute-ABS_HAT0X', 'HX'),
    ('Absolute-ABS_HAT0Y', 'HY'),

    # Face Buttons
    ('Key-BTN_NORTH', 'N'),
    ('Key-BTN_EAST', 'E'),
    ('Key-BTN_SOUTH', 'S'),
    ('Key-BTN_WEST', 'W'),

    # Other buttons
    ('Key-BTN_THUMBL', 'THL'),
    ('Key-BTN_THUMBR', 'THR'),
    ('Key-BTN_TL', 'TL'),
    ('Key-BTN_TR', 'TR'),
    ('Key-BTN_TL2', 'TL2'),
    ('Key-BTN_TR2', 'TR3'),
    ('Key-BTN_MODE', 'M'),
    ('Key-BTN_START', 'ST'),

    # PiHUT SNES style controller buttons
    ('Key-BTN_TRIGGER', 'N'),
    ('Key-BTN_THUMB', 'E'),
    ('Key-BTN_THUMB2', 'S'),
    ('Key-BTN_TOP', 'W'),
    ('Key-BTN_BASE3', 'SL'),
    ('Key-BTN_BASE4', 'ST'),
    ('Key-BTN_TOP2', 'TL'),
    ('Key-BTN_PINKIE', 'TR')
)

# For the Xbox controller, you can set this to 0
MIN_ABS_DIFFERENCE = 0
distracted = 0
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

class JSTest(object):
    def __init__(self, gamepad=None, abbrevs=EVENT_ABB):
        self.btn_state = {}
        self.old_btn_state = {}
        self.abs_state = {}
        self.old_abs_state = {}
        self.abbrevs = dict(abbrevs)
        for key, value in self.abbrevs.items():
            if key.startswith('Absolute'):
                self.abs_state[value] = 0
                self.old_abs_state[value] = 0
            if key.startswith('Key'):
                self.btn_state[value] = 0
                self.old_btn_state[value] = 0
        self._other = 0
        self.gamepad = gamepad
        if not gamepad:
            self._get_gamepad()

    def _get_gamepad(self):
        """Get a gamepad object."""
        try:
            self.gamepad = inputs.devices.gamepads[0]
        except IndexError:
            raise inputs.UnpluggedError("No gamepad found.")

    def handle_unknown_event(self, event, key):
        """Deal with unknown events."""
        if event.ev_type == 'Key':
            new_abbv = 'B' + str(self._other)
            self.btn_state[new_abbv] = 0
            self.old_btn_state[new_abbv] = 0
        elif event.ev_type == 'Absolute':
            new_abbv = 'A' + str(self._other)
            self.abs_state[new_abbv] = 0
            self.old_abs_state[new_abbv] = 0
        else:
            return None

        self.abbrevs[key] = new_abbv
        self._other += 1

        return self.abbrevs[key]

    def process_event(self, event):
        """Process the event into a state."""
        if event.ev_type == 'Sync':
            return
        if event.ev_type == 'Misc':
            return
        key = event.ev_type + '-' + event.code
        try:
            abbv = self.abbrevs[key]
        except KeyError:
            abbv = self.handle_unknown_event(event, key)
            if not abbv:
                return
        if event.ev_type == 'Key':
            self.old_btn_state[abbv] = self.btn_state[abbv]
            self.btn_state[abbv] = event.state
        if event.ev_type == 'Absolute':
            self.old_abs_state[abbv] = self.abs_state[abbv]
            self.abs_state[abbv] = event.state
        self.output_state(event.ev_type, abbv)

    def format_state(self):
        """Format the state."""
        output_string = ""
        for key, value in self.abs_state.items():
            output_string += key + ':' + '{:>4}'.format(str(value) + ' ')

        for key, value in self.btn_state.items():
            output_string += key + ':' + str(value) + ' '

        return output_string

    def output_state(self, ev_type, abbv):
        """Print out the output state."""
        countList = []
        if ev_type == 'Key':
            if self.btn_state[abbv] != self.old_btn_state[abbv]:
                #print(self.format_state())
                buttonList = [int(s) for s in re.findall(r'\b\d+\b', self.format_state())]
                PredictionData = np.array(buttonList)
                PredictionData = PredictionData.reshape(1, -1)
                PredictionData = min_max_scaler.fit_transform(PredictionData.reshape(-1, 1))
                PredictionData = PredictionData.reshape(1, -1)
                X_Predict = PredictionData[:,2:5]
                distracted = ERF.predict_proba(X_Predict)
                print(distracted)
                if distracted[0][0] > .90:
                    countList.append(distracted[0][0])
                    if len(countList) > 1:
                        winsound.Beep(frequency, duration)
                        countList = []
                else:
                    countList = []
                return 
        if abbv[0] == 'H':
            #print(self.format_state())
            buttonList = [int(s) for s in re.findall(r'\b\d+\b', self.format_state())]
            PredictionData = np.array(buttonList)
            PredictionData = PredictionData.reshape(1, -1)
            PredictionData = min_max_scaler.fit_transform(PredictionData.reshape(-1, 1))
            PredictionData = PredictionData.reshape(1, -1)
            X_Predict = PredictionData[:,2:5]
            distracted = ERF.predict_proba(X_Predict)
            print(distracted)
            
            if distracted[0][0] > .90:
                countList.append(distracted[0][0])
                if len(countList) > 1:
                    winsound.Beep(frequency, duration)
                    countList = []
            else:
                countList = []
            return 

        difference = self.abs_state[abbv] - self.old_abs_state[abbv]
        if (abs(difference)) > MIN_ABS_DIFFERENCE:
            #print(self.format_state())
            buttonList = [int(s) for s in re.findall(r'\b\d+\b', self.format_state())]
            PredictionData = np.array(buttonList)
            PredictionData = PredictionData.reshape(1, -1)
            PredictionData = min_max_scaler.fit_transform(PredictionData.reshape(-1, 1))
            PredictionData = PredictionData.reshape(1, -1)
            X_Predict = PredictionData[:,2:5]
            distracted = ERF.predict_proba(X_Predict)
            print(distracted)

            if distracted[0][0] > .90:  
                countList.append(distracted[0][0])
                if len(countList) > 1:
                    winsound.Beep(frequency, duration)
                    countList = []
            else:
                countList = []
            return

    def process_events(self):
        """Process available events."""
        try:
            events = self.gamepad.read()
        except EOFError:
            events = []
        for event in events:
            self.process_event(event)
def main():
    """Process all events forever."""
    jstest = JSTest()
    while 1:
        jstest.process_events()
        

if __name__ == "__main__":
    main()
