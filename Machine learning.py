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
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier

#Reading and formatting distracted test results 
names = ['Speed' , 'Steering input X' , 'Steering input Y']
filenames1 = ['C:\\Users\\Joerg\\Desktop\\Joerg_distracted.csv', 'C:\\Users\\Joerg\\Desktop\\Joerg_distracted_1.csv', 'C:\\Users\\Joerg\\Desktop\\Joerg_distracted_6.csv' , 'C:\\Users\\Joerg\\Desktop\\Joerg_distracted_7.csv', 'C:\\Users\\Joerg\\Desktop\\Joerg_distracted_8.csv' , 'C:\\Users\\Joerg\\Desktop\\Joerg_distracted_9.csv', 'C:\\Users\\Joerg\\Desktop\\Joerg_distracted_10.csv', 'C:\\Users\\Joerg\\Desktop\\Joerg_distracted_11.csv']
data_distracted = [pd.read_csv(f, names = names) for f in filenames1]
data_distracted = pd.concat(data_distracted)
data_distracted['Distracted'] = 1

filenames2 = ['C:\\Users\\Joerg\\Desktop\\Joerg_non_distracted.csv', 'C:\\Users\\Joerg\\Desktop\\Joerg_non_distracted_1.csv','C:\\Users\\Joerg\\Desktop\\Joerg_non_distracted_2.csv', 'C:\\Users\\Joerg\\Desktop\\Joerg_non_distracted_3.csv', 'C:\\Users\\Joerg\\Desktop\\Joerg_non_distracted_4.csv', 'C:\\Users\\Joerg\\Desktop\\Joerg_non_distracted_5.csv', 'C:\\Users\\Joerg\\Desktop\\Joerg_non_distracted_6.csv', 'C:\\Users\\Joerg\\Desktop\\Joerg_non_distracted_7.csv', 'C:\\Users\\Joerg\\Desktop\\Joerg_non_distracted_8.csv', 'C:\\Users\\Joerg\\Desktop\\Joerg_non_distracted_9.csv', 'C:\\Users\\Joerg\\Desktop\\Joerg_non_distracted_10.csv', 'C:\\Users\\Joerg\\Desktop\\Joerg_non_distracted_11.csv' ]
data_non_distracted = [pd.read_csv(f, names = names) for f in filenames2]
data_non_distracted = pd.concat(data_non_distracted)
data_non_distracted['Distracted'] = 0

filenames3 = ['C:\\Users\\Joerg\\Desktop\\Angie_test_data.csv']
PredictionData = [pd.read_csv(f, names = names) for f in filenames3]
PredictionData = pd.concat(PredictionData)

#Creating shrinking function for preprocessing
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

#Preparing prediction data for visualization
PredictionData = PredictionData.values
PredictionData = min_max_scaler.fit_transform(PredictionData)
X_Predict = PredictionData[:,0:3]

#Preparing data for modeling
data = data_distracted.append(data_non_distracted)
array = data.values
array = min_max_scaler.fit_transform(array)
X = array[:,0:3]
Y = array[:,3]
validation_size = 0.18
seed = 7
scoring = 'accuracy'
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Testing to find the best algorithm
models = []
#models.append(('SVM', SVC(cache_size = 3000))) Too resource intensive at higher then quadratic time costs
#models.append(('LR', LogisticRegression()))    #Doesn't match the dataset's distribution 
#models.append(('LDA', LinearDiscriminantAnalysis())) #Doesn't match the dataset's distribution 
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state = 0)))
#models.append(('NB', GaussianNB())) #Doesn't match the dataset's distribution 
#models.append(('RF', RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)))
#models.append(('ERF', ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)))
#models.append(('ADA' , AdaBoostClassifier(DecisionTreeClassifier(max_depth=None))))

#Evaluate each model and print the accuracy scores of each
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=50, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

ERF = ExtraTreesClassifier(n_estimators=300, max_depth=None, min_samples_split=2, random_state=0)
ERF.fit(X, Y)
kfold = model_selection.KFold(n_splits=90, random_state=seed)
cv_results = model_selection.cross_val_score(ERF, X_train, Y_train, cv=kfold, scoring=scoring)
results.append(cv_results)

msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
print(msg)

pickle.dump(ERF, open( "ERFtest.pkl", "wb" ) )