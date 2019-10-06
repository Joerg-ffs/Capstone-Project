import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

#Read csv file
df = pd.read_csv('03_10_2019_A.csv')


# Get column names first
print(df.columns)
names = df.columns

# Create the Scaler object
scaler = preprocessing.StandardScaler()

# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=names)

scaled_df['Distraction'] = 1  #1 = non-distracted

scaled_df = scaled_df[['Angle', 'Prediction', 'Distraction']]

plt.plot(scaled_df['Angle'][8700:8900])
plt.plot(scaled_df['Prediction'][8700:8900])

scaled_df.to_csv("03_10_2019_A_Scaled.csv", sep=',', index=False)

print(scaled_df)

plt.show()