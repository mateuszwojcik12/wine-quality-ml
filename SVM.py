import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

red_wines = pd.read_csv("data.csv")

X = red_wines.iloc[:, :-1]
Y = red_wines.iloc[:, -1:]

X_train, X_test, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size = 0.20)

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, Y_TRAIN.values.ravel())

