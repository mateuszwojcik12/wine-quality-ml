from mlxtend.feature_selection import SequentialFeatureSelector as SBS
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

red_wines = pd.read_csv("data.csv")
X = red_wines.iloc[:, :-1]
Y = red_wines.iloc[:, -1:]

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.25)

knn = KNeighborsClassifier(n_neighbors=4)

sbs = SBS(knn,
          k_features=3,
          forward=False,
          floating=False,
          scoring='accuracy',
          cv=4,
          n_jobs=-1)

sbs = sbs.fit(X_TRAIN, Y_TRAIN.values.ravel())

print('\nSequential Backward Selection (k=3):')
print(sbs.k_feature_idx_)
print('CV Score:')
print(sbs.k_score_)

print(pd.DataFrame.from_dict(sbs.get_metric_dict()).T)