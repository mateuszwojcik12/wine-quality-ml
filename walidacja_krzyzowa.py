from sklearn.model_selection import train_test_split
import pandas as pd

# Dzielenie na podzbiory

red_wines = pd.read_csv("data.csv")
X = red_wines.iloc[:, :-1]
Y = red_wines.iloc[:, -1:]

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.25, random_state=0)

# Nauka

# todo: Tutaj uczy się algorytm

# Test skuteczności
from sklearn import metrics

cm = metrics.confusion_matrix(Y_TEST, y_pred)
print(cm)
accuracy = metrics.accuracy_score(Y_TEST, y_pred)
print("Accuracy score:", accuracy)
precision = metrics.precision_score(Y_TEST, y_pred, average='micro')
print("Precision score:", precision)
recall = metrics.recall_score(Y_TEST, y_pred, average='micro')
print("Recall score:", recall)
