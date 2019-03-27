"""
This program creates visualizations of our datasets
Created by Joerg Oxborough and Matthew Clark
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

np.set_printoptions(threshold=100)

#read in data
data_n = pd.read_csv("non-distracted_2.csv")
data_d = pd.read_csv("distracted_2.csv")
data_angle_n = np.array(data_n['Angle'])
data_pred_n = np.array(data_n['Prediction'])

#get columns needed for visualization
data_angle_d = np.array(data_d['Angle'])
data_pred_d = np.array(data_d['Prediction'])

#scale data
data_angle_d = preprocessing.scale(data_angle_d)
data_pred_d = preprocessing.scale(data_pred_d)

#plot data
plt.plot(data_angle_d)
plt.plot(data_pred_d)
#plt.plot(data_pred_norm)
#plt.scatter()
#plt.plot(data_angle_norm)

#print correlation coefficient
print(pearsonr(data_angle_n, data_pred_n))

#show plot
plt.show()



