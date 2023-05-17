from sklearn.neighbors import NearestCentroid
from scipy.spatial import distance
import numpy as np



list_a = np.array([[0,1]])
list_b = np.array([[0,1]])
dist = distance.cdist(list_a, list_b, 'euclidean')
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
clf = NearestCentroid()
clf.fit(X, y)
a = clf.centroids_
# d = distance.cdist(np.array([X[0]]), np.array([a[0]]), 'euclidean')
d = 0
for i, v in enumerate(X):
    # print(v)
    # print(np.array(a[y[i]-1]))
    d += pow(distance.cdist(np.array([v]), np.array([a[y[i]-1]]), 'euclidean'), 2)


