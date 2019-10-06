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
df = pd.read_csv('comma_random.csv')
img_name = df['Frame']
angle = df['Angle']
Prediction = []
Frames = []

i = 0
for j in img_name:

    try:
        img = cv2.imread(img_path + str(img_name.iloc[i]) + '.jpg', 3)
        pred_img = cv2.resize(np.uint8(img), (240, 120))
        pred_img = pred_img.reshape(-1,120, 240, 3)
        cv2.imshow('test', img)
        cv2.waitKey(1)

        pred = model.predict(pred_img)

        Prediction.append(pred[0][0])
        Frames.append(angle.iloc[i])

    except:
        print(str(img_name.iloc[i]))


    i = i + 1
    if i == 40000:
        df1 = pd.DataFrame(data={"Prediction": Prediction, "Angle": Frames})
        df1.to_csv("03_10_2019_A.csv", sep=',', index=False)
        cv2.waitKey(40)
        break


