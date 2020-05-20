if __name__ == '__main__':
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    import pandas as pd

    dt = DecisionTreeClassifier()

    red_wines = pd.read_csv("data.csv")
    X = red_wines.iloc[:, :-1]
    Y = red_wines.iloc[:, -1:]

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.25)

    dt.fit(X_TRAIN, Y_TRAIN.values.ravel())

    # prognozy
    y_pred = dt.predict(X_TEST)

    from sklearn import metrics

    cm = metrics.confusion_matrix(Y_TEST, y_pred)
    print(cm)
    accuracy = metrics.accuracy_score(Y_TEST, y_pred)
    print("Accuracy score:", accuracy)
    precision = metrics.precision_score(Y_TEST, y_pred, average='micro')
    print("Precision score:", precision)
    recall = metrics.recall_score(Y_TEST, y_pred, average='micro')
    print("Recall score:", recall)
    f1_score = (2*precision*recall)/(precision+recall)
    print("F1 score: " + str(f1_score))
