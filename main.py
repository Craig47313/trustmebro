import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

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
print(raw)
#Defin X,Y & Split
X = pd.concat([raw['Year'], raw['Distance zscore'], raw['Owner'], makes, drives],axis=1)
y = raw['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.linear_model import LinearRegression

rf = LinearRegression()
#Train Model
rf = RandomForestRegressor(random_state=4)
rf.fit(X_train, y_train)


print('train')
preds = rf.predict(X_train)

mse = np.mean(pow(preds - y_train, 2))
print(np.sqrt(mse))
nrmse = np.sqrt(mse) / np.mean(y_train)
print('test')



preds = rf.predict(X_test)

from sklearn.model_selection import KFold, cross_val_score

r2 = cross_val_score(rf, X_test, y_test, cv='kf')


mse = np.mean(pow(preds - y_test, 2))
print(np.sqrt(mse))
nrmse = np.sqrt(mse) / np.mean(y_test)
print(nrmse)

#print(X.head(10))






print(''.join(np.repeat('-=+=-<->', 15)))