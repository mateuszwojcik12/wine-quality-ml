if __name__ == '__main__':
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    import pandas as pd

    knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')

    red_wines = pd.read_csv("data.csv")
    X = red_wines.iloc[:, :-1]
    Y = red_wines.iloc[:, -1:]

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.25)

    knn.fit(X_TRAIN, Y_TRAIN.values.ravel())

    # prognozy
    y_pred = knn.predict(X_TEST)

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


