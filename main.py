import pandas as pd

path = 'cars_24_combined.csv'

raw = pd.read_csv(path)
print(raw.head())