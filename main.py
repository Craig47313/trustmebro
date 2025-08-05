import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

''.join(np.repeat('\n', 10))

path = 'cars_24_combined.csv'

raw = pd.read_csv(path)
print(raw.head(10))

X = pd.concat([raw['Year'], raw['Distance'], raw['Owner'] ],axis=1)
y = raw['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y)







print(''.join(np.repeat('-=+=-<->', 15)))