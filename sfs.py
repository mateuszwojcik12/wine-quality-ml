from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

red_wines = pd.read_csv("data.csv")
X = red_wines.iloc[:, :-1]
Y = red_wines.iloc[:, -1:]

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.25)

knn = KNeighborsClassifier(n_neighbors=4)

sfs = SFS(knn,
          k_features=3,
          forward=True,
          floating=False,
          scoring='accuracy',
          cv=4,
          n_jobs=-1)

sfs = sfs.fit(X_TRAIN, Y_TRAIN.values.ravel())

print('\nSequential Forward Selection (k=3):')
print(sfs.k_feature_idx_)
print('CV Score:')
print(sfs.k_score_)
