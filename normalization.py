from sklearn.preprocessing import MinMaxScaler

import pandas as pd

mms = MinMaxScaler()

df = pd.read_csv('data.csv')

x = df.values

min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

df = pd.DataFrame(x_scaled)

#df.to_csv("data_norm")