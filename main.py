import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print(''.join(np.repeat('\n', 10)))

#Load Dataset
path = 'cars_24_combined.csv'
raw = pd.read_csv(path)

#Remove NaN Values
raw = raw.dropna()

#Get Company Name
raw['make'] = raw['Car Name'].apply(lambda x : x.split(' ')[0])

#One-Hot Encoding
makes = pd.get_dummies(raw.make)
drives = pd.get_dummies(raw.Drive)
raw = pd.concat([raw, makes, drives], axis=1)

#Z-score Normalization
disSTD = np.std(raw['Distance'])
disMU = np.mean(raw['Distance'])
raw['Distance zscore'] = raw['Distance'].apply(lambda x : (x-disMU)/disSTD)

#Defin X,Y & Split
X = pd.concat([raw['Year'], raw['Distance zscore'], raw['Owner'], makes, drives],axis=1)
y = raw['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=4)
rf.fit(X_train, y_train)


print('X test[0]:')
print(X_train.iloc[2])
print('Prediction:')
print(rf.predict(X_train.iloc[[2]]))
print('Answer')
print(y_train.iloc[2])

#print(X.head(10))






print(''.join(np.repeat('-=+=-<->', 15)))