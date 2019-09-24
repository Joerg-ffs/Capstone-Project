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

# Write some Text
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,700)
fontScale = .8
fontColor = (255,255,255)
lineType = 2

img_path = 'C://Users//joerg//PycharmProjects//thirdeye//comma1//frames//'
df = pd.read_csv('video_data.csv')
img_name = df['Frame']
angle = df['Angle']

i = 10
for j in img_name:

    img = cv2.imread(img_path + str(img_name.iloc[i]) +'.jpg')


    try:
        pred_img = np.reshape(img, (-1, 120, 240, 3))
        pred = model.predict(pred_img)

        img = cv2.resize(img, (720 * 2, 360 * 2))
        diff = angle.iloc[i] - pred[0][0]
        cv2.putText(img, 'Actual Angle:' + str(angle.iloc[i]) + ' Predicted:' + str(pred[0][0]) + ' Difference: '+ str(diff),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        cv2.imshow('test', img)
        cv2.waitKey(80)


    except:
        print(str(img_name.iloc[i]))
    i = i + 1

