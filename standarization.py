from sklearn.preprocessing import StandardScaler

import pandas as pd

df = pd.read_csv('data.csv')

x = df.values

stdsc = StandardScaler()
x_scaled = stdsc.fit_transform(x)

df = pd.DataFrame(x_scaled)

#df.to_csv('data.csv')