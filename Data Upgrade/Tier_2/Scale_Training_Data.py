import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

#Read csv file
df = pd.read_csv('training_data.csv')
df1 = df['Distraction']

# Get column names first
print(df.columns)
names = df.drop(['Accelerator', 'Brake', 'Distraction'], axis=1).columns

# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df.drop(['Accelerator','Brake','Distraction'], axis=1))
scaled_df = pd.DataFrame(scaled_df, columns=['Prediction', 'Angle'])

scaled_df['Distraction'] = df1

plt.plot(scaled_df['Angle'][8000:8100])
plt.plot(scaled_df['Prediction'][8000:8100])

scaled_df.to_csv("training_data_Scaled.csv", sep=',', index=False)

plt.show()