if __name__ == '__main__':
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression

    red_wines = pd.read_csv("data.csv")
    X = red_wines.iloc[:, :-1]
    Y = red_wines.iloc[:, -1:]

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.25)

    knn = KNeighborsClassifier(n_neighbors=5)
    params_knn = {'n_neighbors': np.arange(1, 50)}
    knn_gs = GridSearchCV(knn, params_knn, cv=5)
    knn_gs.fit(X_TRAIN, Y_TRAIN.values.ravel())

    # Najlepszy model
    knn_best = knn_gs.best_estimator_
    print(knn_gs.best_params_)

    rf = RandomForestClassifier()
    params_rf = {'n_estimators': [50, 100, 200]}
    rf_gs = GridSearchCV(rf, params_rf, cv=5)
    rf_gs.fit(X_TRAIN, Y_TRAIN.values.ravel())

    rf_best = rf_gs.best_estimator_
    print(rf_gs.best_params_)

    log_reg = LogisticRegression(multi_class='ovr', max_iter=10000)
    log_reg.fit(X_TRAIN, Y_TRAIN.values.ravel())

    svclassifier = SVC()
    params_svm = {'kernel',('linear')}
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_TRAIN, Y_TRAIN.values.ravel())

    dt = DecisionTreeClassifier()
    dt.fit(X_TRAIN, Y_TRAIN.values.ravel())

    print('knn: {}'.format(knn_best.score(X_TEST, Y_TEST.values.ravel())))
    print('rf: {}'.format(rf_best.score(X_TEST, Y_TEST.values.ravel())))
    print('log_reg: {}'.format(log_reg.score(X_TEST, Y_TEST.values.ravel())))
    print('SVM: {}'.format(svclassifier.score(X_TEST, Y_TEST.values.ravel())))
    print('dt: {}'.format(svclassifier.score(X_TEST, Y_TEST.values.ravel())))

    # OvA
    from sklearn.ensemble import VotingClassifier
    estimators = [('knn', knn_best), ('rf', rf_best), ('log_reg', log_reg), ('svm', svclassifier), ('dt', dt)]
    ensemble = VotingClassifier(estimators, voting='hard')
    ensemble.fit(X_TRAIN, Y_TRAIN.values.ravel())
    result = ensemble.score(X_TEST, Y_TEST.values.ravel())

    print(result)