import numpy as np
import pandas as pd
import pickle

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn import model_selection
from sklearn.ensemble import ExtraTreesClassifier

#Splits array l into n sized chunks
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

#Preparing data for modeling
data = pd.read_csv("Scaled_Merged_Data.csv")

data.drop(data.tail(3).index, inplace=True)

#Gets x and y values from columns
data_y = data['Distraction']
data_x = data.drop("Distraction", axis=1)

print(len(data_x))

#Split arrays into arrays of 100 size (262303 data points / 100 = 2623)
X = np.array_split(data_x, 2623)
y = np.array_split(data_y, 2623)

j = 0
#Makes each datapoint an array
for i in X:
  X[j] = np.array(X[j])
  y[j] = np.array(y[j])
  j += 1

X = np.asarray(X)
y = np.asarray(y)
X = X.reshape(2623, 200) #flattens 2D 59x4 chunks to 1D 236 chunks
y = y.reshape(2623, 100) #flattens 1D 59x1 chunks to 1D 59 chunks

#Model parameters
validation_size = 0.15
seed = 7
scoring = 'accuracy'
results = []
#splits X and y into training and validation sets
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)

#model definition
ERF = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
#fit to dataset
ERF.fit(X, y)
kfold = model_selection.KFold(n_splits=90, random_state=seed)
#score model
cv_results = model_selection.cross_val_score(ERF, X_train, Y_train, cv=kfold, scoring=scoring)
results.append(cv_results)

#print score
msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
print(msg)

#dump pickle of model
pickle.dump(ERF, open( "Tier_2_Model.pkl", "wb" ) )