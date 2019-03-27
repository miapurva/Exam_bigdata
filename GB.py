from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification

import pandas as pd
mydata = pd.read_csv("employee_reviews.csv")
target = mydata["senior_mangemnet_stars"]
#print target
X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
clf = GradientBoostingClassifier(n_estimators=100, max_depth=2,
                             random_state=0)
clf.fit(X, y)

print(clf.feature_importances_)
#print(clf.fit(X, y))
print(clf.predict([[0, 0, 0, 0]]))