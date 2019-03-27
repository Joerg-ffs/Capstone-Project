import numpy as np
import pandas as pd
import os
import cv2
import random
import pickle
import gzip
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats.stats import pearsonr

np.set_printoptions(threshold=100)

data_n = pd.read_csv("non-distracted_2.csv")
data_d = pd.read_csv("distracted_2.csv")
data_angle_n = np.array(data_n['Angle'])
data_pred_n = np.array(data_n['Prediction'])

data_angle_d = np.array(data_d['Angle'])
data_pred_d = np.array(data_d['Prediction'])

data_angle_d = preprocessing.scale(data_angle_d)
data_pred_d = preprocessing.scale(data_pred_d)
plt.plot(data_angle_d)
plt.plot(data_pred_d)
#plt.plot(data_pred_norm)
#plt.scatter()
#plt.plot(data_angle_norm)

print(pearsonr(data_angle_n, data_pred_n))

plt.show()



