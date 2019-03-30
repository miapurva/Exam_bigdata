X = df.copy()
X = X.drop('IS_TRAFFIC', axis = 1)
y = df['IS_TRAFFIC']

def holdout_cv(X,y,size = 0.3, seed = 1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = seed)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = holdout_cv(X, y, size = 0.3, seed = 1)