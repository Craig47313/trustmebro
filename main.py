import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print(''.join(np.repeat('\n', 10)))

path = 'cars_24_combined.csv'

raw = pd.read_csv(path)
#print(raw.head(10))

raw = raw.dropna()

raw['make'] = raw['Car Name'].apply(lambda x : x.split(' ')[0])
#print(raw['make'])

makes = pd.get_dummies(raw.make)
drives = pd.get_dummies(raw.Drive)
#print(makes)

raw = pd.concat([raw, makes, drives], axis=1)

X = pd.concat([raw['Year'], raw['Distance'], raw['Owner'], makes, drives],axis=1)
y = raw['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y)

'''
Data-prep:
- Remove nan values
'''

print(X.head(10))




print(''.join(np.repeat('-=+=-<->', 15)))