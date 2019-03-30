from sklearn import metrics
from collections import Counter
from scipy.spatial.distance import pdist,squareform

# Any results you write to the current directory are saved as output.

import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
import apriori   
from tqdm import tqdm

store_data = pd.read_csv('yutika.csv')  

records = []  
for i in tqdm(range(100)):  
    records.append([str(store_data.values[i,j]) for j in range(0, 100)])


association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)  
association_results = list(association_rules) 
print(len(association_results)) 

print(association_results[0])  
print("Yutika")

for item in association_results:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    confidence =  item[2][0][2]
    lift =  item[2][0][3]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    support_AC = item[1]
    support_A = support_AC / confidence
    support_C = confidence / lift
    
    leverage = support_AC - support_A*support_C
    conviction = (1 - support_C) / (1 - confidence)
    
    print("Support: YUTIKA KULWE " , support_AC)
    print("Confidence: " , confidence)
    print("Lift: " ,lift)
    print("Leverage: " , leverage)
    print("Conviction : " , conviction)
    print("=====================================")