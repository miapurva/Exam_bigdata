import matplotlib.pyplot as plt  
import numpy as np  
from sklearn.cluster import KMeans  
from sklearn import metrics
from collections import Counter
from scipy.spatial.distance import pdist,squareform
import random
X = np.array([[random.randint(0,67530),random.randint(0,67530)],  
     [random.randint(0,67530),random.randint(0,67530)],
     [random.randint(0,67530),random.randint(0,67530)],
     [random.randint(0,67530),random.randint(0,67530)],
     [random.randint(0,67530),random.randint(0,67530)],
     [random.randint(0,67530),random.randint(0,67530)],
     [random.randint(0,67530),random.randint(0,67530)],
     [random.randint(0,67530),random.randint(0,67530)],
     [random.randint(0,67530),random.randint(0,67530)],
     [random.randint(0,67530),random.randint(0,67530)],])

true_clusters_labels = [0,0,0,0,0,1,1,1,1,1]

plt.scatter(X[:,0],X[:,1], label='True Position') 
kmeans = KMeans(n_clusters=3)  
distances = kmeans.fit_transform(X) 
print("=================Distances==================") 
print(distances)
plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')  
print("=================Cluster centres==================")
print(kmeans.cluster_centers_)  
print("=================Labels==================")
print(kmeans.labels_)
predicted_labels = kmeans.labels_
num_of_members = Counter(predicted_labels)
cluster_radius = {x:0 for x in predicted_labels}
for distance,label in zip(distances,predicted_labels):
    cluster_radius[label] += distance[label]

for label in num_of_members:
    cluster_radius[label] /= num_of_members[label]
print('Cluster Radius : ',cluster_radius)


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    print("=================Contingency Matrix==================")
    print(contingency_matrix)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 


purity_score(true_clusters_labels,predicted_labels)	