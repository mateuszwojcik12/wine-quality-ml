import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

red_wines = pd.read_csv("data.csv")

X = red_wines.iloc[:, :-1]
Y = red_wines.iloc[:, -1:]

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.25)

model = LogisticRegression(multi_class='ovr', max_iter=10000)

model.fit(X_TRAIN, Y_TRAIN.values.ravel())

score = model.score(X_TEST, Y_TEST)

print(score)