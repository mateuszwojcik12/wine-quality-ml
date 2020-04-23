from sklearn.impute import KNNImputer

import pandas as pd

data = pd.read_csv('data.csv')

#Deklaracja Imputer-a, parameter n_neighbors oznacza ilość sąsiadów
imputer = KNNImputer(n_neighbors=2)

#uzupełnianie danych brakujących
df_filled = imputer.fit_transform(data)

#df.to_csv('data.csv')