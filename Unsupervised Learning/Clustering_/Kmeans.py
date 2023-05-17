import numpy as np
import pandas as pd

data = pd.read_csv('test1.csv', usecols=[1, 2, 3, 4])
data = np.array(data)
print("raw data: ", np.shape(data))

# Calculate distance between data
def ditance(a,b):
    dis = 0
    for i in range(3):
        dis += (a[i]-b[i])**2
    return np.sqrt(np.sum(dis))

# Randomly pick the initial k centres
def initial(dataset,k):
    centres_i = []
    for i in range(k):
        centres_i.append(dataset[int(len(dataset)*np.random.rand()//1)])
    return np.array(centres_i)

def kmeans(dataset,k,num):
    # store the centres of each cluster
    centres_k = initial(dataset, k)
    m = len(dataset)
    n = len(dataset[0])

    # store information of each data in the form of (labels of centres, distance to the centres)
    info = np.zeros((m, 2))
    index = 0
    while index < num:
        index = index+1
        for i in range(m):
            mindis = np.inf
            label = -1
            for j in range(k):
                dis = ditance(dataset[i], centres_k[j])
                if dis < mindis:
                    mindis = dis
                    label = j
            info[i] = label, mindis
        for i in range(k):
            fz = np.nonzero(info[:, 0] == i)[0]
            centres_k[i] = np.mean(dataset[fz], axis=0)
    return centres_k, info

# set number of clusters and operations
centres, info = kmeans(data, 5, 10**2)

labelzero = np.where(info == 0)[0]
lbone = np.where(info == 1)[0]
lbtwo = np.where(info == 2)[0]
lbthree = np.where(info == 3)[0]
lbfour = np.where(info == 4)[0]
lbfive = np.where(info == 5)[0]

print("cluster 0:", np.shape(labelzero), "cluster 0:", np.shape(lbone), "cluster 2:", np.shape(lbtwo),
      "cluster 3:", np.shape(lbthree), "cluster 4:", np.shape(lbfour))

rawdata = pd.read_csv('test1.csv', usecols=[1, 2, 3, 4])

# select clusters to be dropped
droplabels = np.concatenate([lbtwo, lbthree])
print("number of data to be dropped:", len(droplabels))
rawdata = rawdata.drop(droplabels)
print("new data: ", np.shape(rawdata))
headerList = ["time", "SF_x", "SF_y", "SF_z"]
np.savetxt("test1new_kmeans.csv", rawdata, delimiter=',', fmt="%s")
file = pd.read_csv("test1new_kmeans.csv")
file.to_csv("test1new_kmeans.csv", header=headerList, index=False)