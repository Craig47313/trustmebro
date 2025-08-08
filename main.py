import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor

print(''.join(np.repeat('\n', 10)))

#Load Dataset
path = 'cars_24_combined.csv'
raw = pd.read_csv(path)

#Remove NaN Values
raw = raw.dropna()
print(raw)

#Get Company Name
raw['make'] = raw['Car Name'].apply(lambda x : x.split(' ')[0])

#One-Hot Encoding
makes = pd.get_dummies(raw.make)
drives = pd.get_dummies(raw.Drive)
fuels = pd.get_dummies(raw.Fuel)
locs = pd.get_dummies(raw.Location)
types = pd.get_dummies(raw.Type)
raw = pd.concat([raw, makes, drives, locs, types, fuels], axis=1)

#Z-score Normalization
disSTD = np.std(raw['Distance'])
disMU = np.mean(raw['Distance'])
raw['Distance zscore'] = raw['Distance'].apply(lambda x : (x-disMU)/disSTD)
raw['Age'] = raw['Year'].apply(lambda x : 2025 - x)
print(raw)

#Defin X,Y & Split
X = pd.concat([raw['Age'], raw['Distance zscore'], raw['Owner'], locs, makes, drives, types, fuels],axis=1)
y = raw['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y)

#Train Model
rf = RandomForestRegressor(random_state=4)
rf.fit(X_train, y_train)
preds = rf.predict(X_train)
preds = rf.predict(X_test)

#Evaluate Model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse = -cross_val_score(rf, X_test, y_test,cv=kf, scoring='neg_mean_squared_error')
r2 = cross_val_score(rf, X_test, y_test, cv=kf, scoring='r2')
print(f'The MSE = {mse.mean():.2f}(±{mse.std():.2f})')
print(f'The RMSE = {np.sqrt(mse.mean()):.2f}')
print(f'The N-RMSE = {np.sqrt(mse.mean()) / np.mean(y_test):.2f}')
print(f'The R2 = {r2.mean():.4f}(±{r2.std():.4f})')
print(''.join(np.repeat('-=+=-<->', 15)))
