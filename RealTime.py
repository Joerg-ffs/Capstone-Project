#importing libraries to read, plot, and manipulate the data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
count = 0
filenames3 = ['C:\\Users\\Joerg\\Desktop\\Angie_test_data.csv']
PredictionData = [pd.read_csv(f) for f in filenames3]
PredictionData = pd.concat(PredictionData)

#Creating shrinking function for preprocessing
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
#Preparing prediction data for visualization
PredictionData = PredictionData.values
PredictionData = min_max_scaler.fit_transform(PredictionData)
X_Predict = PredictionData[:,0:3]

ERF = pickle.load( open('ERFtest.pkl', 'rb'))
distracted = ERF.predict_proba(X_Predict)
DisProb, NonProb = zip(*distracted)
plt.plot(DisProb)
plt.show()