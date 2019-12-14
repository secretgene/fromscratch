# clustering dataset
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


n = int(input('Enter the number of points you need to cluster: '))
arr1 = []
arr2 = []
for i in range(n):
    x = int(input('Enter the value of X: '))
    arr1.append(x)
    y = int(input('Enter the value of Y: '))    
    arr2.append(y)
    
x1 = np.array(arr1)
x2 = np.array(arr2)
plt.plot()
plt.xlim([0, 20])
plt.ylim([0, 20])
plt.title('Dataset')
plt.scatter(x1, x2)
plt.show()

# create new plot and data
plt.plot()
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
colors = ['b', 'g', 'r', 'y', 'p']
markers = ['o', 'v', 's', 'H', '+']

# KMeans algorithm 

K = int(input('Enter the value for K: '))
kmeans_model = KMeans(n_clusters=K).fit(X)

plt.plot()
for i, l in enumerate(kmeans_model.labels_):
    plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l],ls='None')
    plt.xlim([0, 20])
    plt.ylim([0, 20])

plt.show()
