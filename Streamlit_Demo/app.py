import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os, urllib, cv2
import time

import cv2
import pandas as pd
import numpy as np
import keras
from keras.models import load_model
from keras import backend as K

def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))

def keras_predict(model, image):
    processed = keras_process_image(image)
    steering_angle = float(model.predict(processed, batch_size=1))
    # steering_angle *= 100
    return steering_angle

def keras_process_image(img):
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, 120, 240, 3))
    return img

model = load_model("Model_3.h5", custom_objects={'coeff_determination': coeff_determination})

img_path = 'C://Users//joerg//PycharmProjects//thirdeye//comma1//frames//'
#df = pd.read_csv('comma_random.csv')
df = pd.read_csv('video_data.csv')
img_name = df['Frame']
angle = df['Angle']
Prediction = []
Frames = []
Prediction_graph = []
Frames_graph = []
display_img = []
caption_list = []
chart_data = pd.DataFrame(columns=['Prediction', 'Actual'])

i = 0

st.title('Test Images')
imageLocation = st.empty()
predictionLocation = st.empty()
actualLocation = st.empty()
differenceLocation = st.empty()

for j in img_name:
    try:
        img = cv2.imread(img_path + str(img_name.iloc[i]) + '.jpg', 3)
        pred_img = cv2.resize(np.uint8(img), (240, 120))
        display_img.append(pred_img)

        pred_img = pred_img.reshape(-1,120, 240, 3)
        pred = model.predict(pred_img)

        Prediction.append(pred[0][0])
        Prediction_graph.append(Prediction)
        Frames.append(angle.iloc[i])
        Frames_graph.append(Frames)

        #caption_list.append('Prediction: ' + str(Prediction[0]) + ', Actual: ' +str(Frames[0]) +' , Difference: ' + str(Prediction[0] - Frames[0]))
    except:
        print(str(img_name.iloc[i]))
        print(len(Prediction))

    i = i + 1
    if len(Prediction) == 2:
        display_img.pop(0)
        Prediction.pop(0)
        Frames.pop(0)
        #caption_list.pop(0)
    if i == 1:
        chart = st.line_chart(pred - angle.iloc[i])


    chart.add_rows(pred -angle.iloc[i])
    imageLocation.image(display_img, width=615)
    predictionLocation.markdown('Prediction Angle: ' + str(round(Prediction[0], 3)) + '°')
    actualLocation.markdown('Actual Angle: ' + str(round(Frames[0], 3)) + '°')
    differenceLocation.markdown('Difference: ' + str(round((Prediction[0] - Frames[0]), 3)) + '°')

    time.sleep(.01)
