import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import AgglomerativeClustering

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    return linkage_matrix


data = pd.read_csv('test1.csv', usecols=[2, 3, 4])
rawdata = pd.read_csv('test1.csv',usecols=[1, 2, 3, 4])
print("raw data: ", np.shape(rawdata))
# setting distance_threshold=0 ensures we compute the full tree.MS
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(data)
plt.figure(figsize=(10, 6))
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
linkage = plot_dendrogram(model, truncate_mode='level', p=3)
# determine the number of clusters
clusters = fcluster(linkage, 5, criterion='maxclust')
# print(clusters)

lbone = np.where(clusters == 1)[0]
lbtwo = np.where(clusters == 2)[0]
lbthree = np.where(clusters == 3)[0]
lbfour = np.where(clusters == 4)[0]
lbfive = np.where(clusters == 5)[0]

print("cluster 1:", np.shape(lbone), "cluster 2:", np.shape(lbtwo), "cluster 3:", np.shape(lbthree),
      "cluster 4:", np.shape(lbfour), "cluster 5:", np.shape(lbfive))

# select clusters to be dropped
droplabels = np.concatenate([lbtwo, lbfive])
print("number of data to be dropped:", len(droplabels))
rawdata = rawdata.drop(droplabels)
newdata = np.array(rawdata)

print("new data: ", np.shape(newdata))

headerList = ["time", "SF_x", "SF_y", "SF_z"]
np.savetxt("test1new_hierarchical.csv", newdata, delimiter=',', fmt="%s")
file = pd.read_csv("test1new_hierarchical.csv")
file.to_csv("test1new_hierarchical.csv", header=headerList, index=False)

plt.xlabel("Number of points in each node.")
plt.show()
