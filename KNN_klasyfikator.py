if __name__ == '__main__':
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from mlxtend.feature_selection import SequentialFeatureSelector as SFS
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.decomposition import PCA
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier



    red_wines = pd.read_csv("data.csv")
    X = red_wines.iloc[:, :-1]
    Y = red_wines.iloc[:, -1:]

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.25)

    rf = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)

    pca = PCA(n_components=5)
    X_TRAIN = pca.fit_transform(X_TRAIN)
    X_TEST = pca.transform(X_TEST)

    rf.fit(X_TRAIN, Y_TRAIN.values.ravel())

    # dt = DecisionTreeClassifier()

    # sfs1 = SFS(dt,
    #            k_features=3,
    #            forward=True,
    #            floating=False,
    #            verbose=2,
    #            scoring='accuracy',
    #            cv=0)
    #
    # sfs1 = sfs1.fit(X, Y)

    # knn = KNeighborsClassifier(n_neighbors=5)
    #
    # red_wines = pd.read_csv("data.csv")
    # X = red_wines.iloc[:, :-1]
    # Y = red_wines.iloc[:, -1:]
    #
    # X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.25)
    #
    # knn.fit(X_TRAIN, Y_TRAIN.values.ravel())
    #
    # prognozy
    y_pred = rf.predict(X_TEST)

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


