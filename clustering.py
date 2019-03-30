import matplotlib.pyplot as plt

import pandas as pd  
import numpy as np 

import csv
with open('employee_reviews.csv', 'r') as f:
    reader = csv.reader(f)
    customer_data = list(reader)

#customer_data = pd.read_csv("/home/apurva/Desktop/Exam/train.csv")
#customer_data.shape

data = customer_data.iloc[:-15].values  
import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(data, method='ward'))  

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  
cluster.fit_predict(data)   

print(cluster.labels_)  
plt.figure(figsize=(10, 7))  
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')