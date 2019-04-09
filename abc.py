import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
from apyori import apriori  
from sklearn import metrics
from collections import Counter
from scipy.spatial.distance import pdist,squareform

store_data = pd.read_csv('store_data.csv', header=None)

#store_data.head() 

records = []  
for i in range(0, 7501):  
    records.append([str(store_data.values[i,j]) for j in range(0, 20)])

association_rules = apriori(records, min_support=0.045, min_confidence=0.2,min_lift)  
association_results = list(association_rules)  

print(len(list(association_rules)))
print(association_results)
"""
for item in association_rules:
    print("Yutika")
    pair = item[0]

    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))
    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")"""