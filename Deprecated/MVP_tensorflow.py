import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os 
import cv2
import random
import pickle
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, BatchNormalization, Conv2D, Lambda, Dropout, MaxPooling2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import time

#NAME = 'MVP_TESTING-{}'.format(int(time.time()))
#tensorboard = TensorBoard(Logdir='logs/{}'.format(NAME))

DATADIR = "C:/Users/Joerg/Desktop/openxc-wheel-master" #path to frames folder
folder = "frames"                    #folder holding the img files
path = os.path.join(DATADIR, folder) #setting path variable
IMG_HEIGHT = 50                      #Image Height in pixels (around 30 x 30 to 50 x 50 is ideal)
IMG_WIDTH = 100                      #image width in pixels  (around 30 x 30 to 50 x 50 is ideal)
training_data = []
data = pd.read_csv("test_2_csv.csv") #csv file holding angle values

def create_training_data(): #Used to comb through a folder pairing img data and angle data into a list
    i = 0
    for img in os.listdir(path):
        i = i +1                     #iterating through csv file to pair angle values to imgs
        angle = data.iloc[i, 0]
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) #creating an array of grayscale imgs
        new_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))           #Scaling the imgs into new_array
        training_data.append([new_array, angle])                             #Adding imgs and angles into training_data array
       

#create_training_data() 
#random.shuffle(training_data)

X = []
y = []

#for features, label in training_data:
    #X.append(features)
    #y.append(label)
#X = np.array(X).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1) #Trailing 1 represents 1 dimensions due to grayscale -> 3 for colour

#Saving dataset using pickle
#pickle_out = open("X.pickle", "wb")
#pickle.dump(X, pickle_out)
#pickle_out.close()

#pickle_out = open("y.pickle", "wb")
#pickle.dump(y, pickle_out)
#pickle_out.close()

#Opening saved dataset from pickle
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

#X = np.array(X).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1) #Trailing 1 represents 1 dimensions due to grayscale -> 3 for colour
X = X/255.0

#PilotNet Model will relook at later to debug

model = Sequential()

model.add(Conv2D(24, (3,3), input_shape=X.shape[1:]))
model.add(BatchNormalization())
model.add(Conv2D(36, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(48, (3,3),  activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(50, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1))

sgd = RMSprop(lr=1e-3)
model.compile(optimizer=sgd, loss="mse", metrics=['mae', 'acc'])
model.fit(np.array(X), np.array(y), batch_size=10, epochs=3, validation_split=0.2)
print(model.summary())
