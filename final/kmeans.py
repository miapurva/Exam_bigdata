import matplotlib.pyplot as plt
import seaborn as sns
#import the data, specify data types
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle 
#import mglearn
import time
from nltk.tokenize import TweetTokenizer # doesn't split at apostrophes
import nltk
from nltk import Text
from nltk.tokenize import regexp_tokenize
from nltk.tokenize import word_tokenize  
from nltk.tokenize import sent_tokenize 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

df=pd.read_csv('employee_reviews.csv')
df.dropna(inplace=True)

from pandas.api.types import CategoricalDtype 
  
labels, uniques = pd.factorize(df['summary']) 
  
print("Numeric Representation : \n", labels) 
print("Unique Values : \n", uniques) 

#X = df.copy()
X = labels
print(X)
y, z = pd.factorize(df['company'])
X, y = X.tolist(), y.tolist()
print(X, y)
x = [[0 for i in range(2)] for j in range(len(X))]
for i in range(len(X)):
	x[i][0] = X[i]
	x[i][1] = y[i]
print(x)

from sklearn.cluster import KMeans
import numpy as np
#X = np.array([[1, 2], [1, 4], [1, 0],[10, 2], [10, 4], [10, 0]])
estimator=KMeans(n_clusters=3)
#kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
#kmeans.labels_
y_kmeans=estimator.fit_predict(x)

for i in range(1,len(y_kmeans)+1):
	print("the",i,"went to class",y_kmeans[i-1])