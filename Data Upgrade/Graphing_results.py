import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\joerg\\PycharmProjects\\thirdeye\\test.csv')

actual = df[(df['Actual'] >= -8) & (df['Actual'] <= 8)]
Aug5 = actual['Aug5']
actual = actual['Actual']

print(len(actual))
print(len(Aug5))

r2 = actual.corr(Aug5)

plt.title('Predicted vs Actual Angles. R^2: '+str(r2))
plt.xlabel('Data points')
plt.ylabel('Angle')
plt.plot(actual[0:400])
plt.plot(Aug5[0:400])
plt.show()


print(r2)