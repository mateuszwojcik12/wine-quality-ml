from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
red_wines = pd.read_csv("data.csv")
    X = red_wines.iloc[:, :-1]
    Y = red_wines.iloc[:, -1:]

#     X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.25)
#
# rf.fit(X_TRAIN, Y_TRAIN)