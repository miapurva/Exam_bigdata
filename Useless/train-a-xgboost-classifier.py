# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("employee_reviews.csv"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import SMOTE
from imblearn.metrics import classification_report_imbalanced
import xgboost as xgb
from xgboost import XGBClassifier
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import time

"""
(284807, 31)

1. ----
Index(['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
       'Class'],
      dtype='object')

2. ----
All columns are float except the 'class' which is 0/1 int

3. ----
No Missing Value

4. ----
print(df['Class'].value_counts())
print(df['Class'].value_counts()[0]/len(df['Class']) * 100)
print(df['Class'].value_counts()[1]/len(df['Class']) * 100)
0    284315
1       492
Name: Class, dtype: int64
99.82725143693798
0.1727485630620034
"""

# DATA_File = os.path.join(os.path.dirname(__file__), os.path.pardir, 'data', 'creditcard.csv')
DATA_File = os.listdir('../input')[0]

def data_processor():
    df = pd.read_csv('../input/creditcard.csv')
    df = df.drop('Time', axis=1)
    features_to_select = ['V14', 'V4', 'V17', 'V10', 'V12', 'V20', 'Amount', 'V21', 'V26', 'V28', 'V11', 'V19', 'V8', 'V7', 'V13']

    print('Use StandardScaler to process the column data')
    scaler = StandardScaler()
    df[df.columns[:-1].tolist()] = scaler.fit_transform(df[df.columns[:-1].tolist()])
    # print(df.head(5))
    X = df[df.columns[:-1].tolist()]
    X = X[features_to_select]
    y = df[df.columns[-1]]

    print("Train Test Split ratio is 0.3")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    print('X_train shape: {X_train.shape}')
    print('y_train shape: {y_train.shape}')

    return X_train, X_test, y_train, y_test

def xgb_classifier(X_train, X_test, y_train, y_test, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    """
    关于现在这个模型
    准确率 : 0.9995
    AUC : 0.887708
    F1 Score : 0.847584
    ----------------------------------->
    关于现在这个模型
    准确率 : 0.9996
    AUC 得分 (训练集): 0.977480
    F1 Score 得分 (训练集): 0.858209
    ---------------------------------->
    关于现在这个模型
    ['V14', 'V4', 'V17', 'V10', 'V12', 'V20', 'Amount', 'V21', 'V26', 'V28', 'V11', 'V19', 'V8', 'V7', 'V13']
    准确率 : 0.9996
    AUC 得分 (训练集): 0.978563
    F1 Score 得分 (训练集): 0.859259
    ---------------------------------->
    # {'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 3} 0.862920874517388
    # {'colsample_bytree': 1.0, 'gamma': 0.2} 0.871
    # {'gamma': 0.2, 'scale_pos_weight': 1} 0.8702009952422571
    # {'subsample': 0.6} 0.864310306628855
    """
    alg = XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                        min_child_weight=3, gamma=0.2, subsample=0.6, colsample_bytree=1.0,
                        objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)

    if useTrainCV:
        print("Start Feeding Data")
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train.values, label=y_train.values)
        # xgtest = xgb.DMatrix(X_test.values, label=y_test.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # 建模
    print('Start Training')
    alg.fit(X_train, y_train, eval_metric='auc')

    # param_test1 = {}
    # gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
    #                                                 min_child_weight=3, gamma=0.2, subsample=0.8,
    #                                                 colsample_bytree=1.0,
    #                                                 objective='binary:logistic', nthread=4, scale_pos_weight=1,
    #                                                 seed=27),
    #                         param_grid=param_test1,
    #                         scoring='f1',
    #                         n_jobs=4, iid=False, cv=5)
    # gsearch1.fit(X_train, y_train)
    # print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)

    # 对训练集预测
    print("Start Predicting")
    predictions = alg.predict(X_test)
    pred_proba = alg.predict_proba(X_test)[:, 1]

    # 输出模型的一些结果
    print("\n关于现在这个模型")
    print("准确率 : %.4g" % metrics.accuracy_score(y_test, predictions))
    print("AUC 得分 (训练集): %f" % metrics.roc_auc_score(y_test, pred_proba))
    print("F1 Score 得分 (训练集): %f" % metrics.f1_score(y_test, predictions))

    feat_imp = alg.feature_importances_
    feat = X_train.columns.tolist()
    # clf.best_estimator_.booster().get_fscore()
    res_df = pd.DataFrame({'Features': feat, 'Importance': feat_imp}).sort_values(by='Importance', ascending=False)
    res_df.plot('Features', 'Importance', kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()
    print(res_df)
    print(res_df["Features"].tolist())


def logistic_regression():
    """
    F1 score is: 0.7285714285714285
    AUC Score is: 0.9667565771367231
    """
    X_train, X_test, y_train, y_test = data_processor()
    clf = LogisticRegression(C=1e5)
    clf.fit(X_train, y_train)

    print("Score: ", clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    print("F1 score is: {}".format(f1_score(y_test, y_pred)))
    print("AUC Score is: {}".format(roc_auc_score(y_test, y_pred_proba)))


def logistic_with_smote():
    X_train, X_test, y_train, y_test = data_processor()

    clf = LogisticRegression(C=1e5)
    clf.fit(X_train, y_train)
    # build model with SMOTE imblearn
    smote_pipeline = make_pipeline_imb(SMOTE(random_state=4), clf)

    smote_model = smote_pipeline.fit(X_train, y_train)
    smote_prediction = smote_model.predict(X_test)
    smote_prediction_proba = smote_model.predict_proba(X_test)[:, 1]

    print(classification_report_imbalanced(y_test, smote_prediction))
    print('SMOTE Pipeline Score {}'.format(smote_pipeline.score(X_test, y_test)))
    print("SMOTE AUC score: ", roc_auc_score(y_test, smote_prediction_proba))
    print("SMOTE F1 Score: ", f1_score(y_test, smote_prediction))


def randomForest():
    """
    F1 score is: 0.7857142857142857
    AUC Score is: 0.9450972761670293
    """
    X_train, X_test, y_train, y_test = data_processor()
    # parameters = {'n_estimators': [10, 20, 30, 50], 'max_depth': [2, 3, 4]}

    clf = RandomForestClassifier(max_depth=4, n_estimators=20)
    # clf = GridSearchCV(alg, parameters, n_jobs=4)
    clf.fit(X_train, y_train)
    print("Score: ", clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    print("F1 score is: {}".format(f1_score(y_test, y_pred)))
    print("AUC Score is: {}".format(roc_auc_score(y_test, y_pred_proba)))

    # print("The Features Importance are: ")  # for feature, value in zip(X_train.columns, clf.feature_importances_):
    #     print(feature, value)
    # print(clf.best_estimator_)
    # print(clf.best_params_)
    # print(clf.best_score_)


def neural_nets():
    """
    Score:  0.9994148145547324
    F1 score is: 0.822695035460993
    AUC Score is: 0.9608730286337007
    """
    X_train, X_test, y_train, y_test = data_processor()
    clf = MLPClassifier(hidden_layer_sizes=(100, 100, 100,))

    clf.fit(X_train, y_train)
    print("Score: ", clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    print("F1 score is: {}".format(f1_score(y_test, y_pred)))
    print("AUC Score is: {}".format(roc_auc_score(y_test, y_pred_proba)))


if __name__ == "__main__":
    # logistic_regression()
    # randomForest()
    # logistic_with_smote()
    # neural_nets()
    start = time.time()
    X_train, X_test, y_train, y_test = data_processor()
    xgb_classifier(X_train, X_test, y_train, y_test)
    print("Total Time is: ", (time.time() - start)/60)