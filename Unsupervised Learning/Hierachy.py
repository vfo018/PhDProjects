import numpy as np
import scipy.cluster.hierarchy as hcluster
from sklearn.neighbors import NearestCentroid
import pandas as pd
from HPW_PSNR import hpw_psnr
from matplotlib import pyplot as plt
from PCA import pca
from scipy.spatial import distance
from sklearn import metrics


my_data = pd.read_table('texture - LD - output.txt')
data_set = my_data.iloc[:, 1:4].values
X_test_set = my_data.iloc[1000:1500, 1:4].values
X = my_data.iloc[1000:1500, 1:4].values
# n_clusters = 14
# linkage = hcluster.linkage(X_test_set, metric='euclidean', method='average')
# labels = hcluster.fcluster(linkage, n_clusters, criterion='maxclust') - 1
# clf = NearestCentroid()
# clf.fit(X, labels)
# centroids = clf.centroids_
# samplesCentroids = centroids[labels]
# count_label = np.bincount(labels)
# sort_index = np.argsort(-count_label)
# per = 4/14
# for i, v in enumerate(X):
#     if labels[i] in sort_index[int(n_clusters * per):]:
#         X[i] = samplesCentroids[i][:]
# hpw_hierarchical_f = hpw_psnr(X_test_set, X, 0.1, 1)
# r_hierarchical = max(count_label)/500
# np.save('hpw_hierarchical', hpw_hierarchical_f)

K = range(2, 31)
Sum_of_squared_distances = []
# Compute and plot first dendrogram.
for k in K:
    linkage = hcluster.linkage(X_test_set, metric='euclidean', method='average')
    hc = hcluster.fcluster(linkage, k, criterion='maxclust')
    clf = NearestCentroid()
    clf.fit(X, hc)
    a = clf.centroids_
    d = 0
    for i, v in enumerate(X_test_set):
        # print(v)
        # print(np.array(a[y[i]-1]))
        d += pow(distance.cdist(np.array([v]), np.array([a[hc[i] - 1]]), 'euclidean'), 2)

    Sum_of_squared_distances.append(d[0])
# linkage = hcluster.linkage(X_test_set, metric='euclidean', method='average')


# SC, CHI, DBI = [], [], []
# K = range(10, 21)
# for k in K:
#     linkage = hcluster.linkage(X_test_set, metric='euclidean', method='average')
#     labels = hcluster.fcluster(linkage, k, criterion='maxclust')
#     # clf = NearestCentroid()
#     #     clf.fit(X, hc)
#     sc = metrics.silhouette_score(X_test_set, labels, metric='euclidean')
#     chi = metrics.calinski_harabasz_score(X_test_set, labels)
#     dbi = metrics.davies_bouldin_score(X_test_set, labels)
#     SC.append(sc)
#     CHI.append(chi)
#     DBI.append(dbi)
# #
# fig = plt.figure()
#
# ax1 = fig.add_subplot(111)
# ax1.plot(K, SC, 'bx-', label = 'SC')
# ax1.plot(K, DBI, 'gx-', label = 'DBI')
# ax1.legend(loc=5, fontsize=10)
# scmax = max(SC)
# scpos = SC.index(scmax)
# kmax = K[scpos]
# dbimin = min(DBI)
# dbipos = DBI.index(dbimin)
# kmin = K[dbipos]

# dbmax = max(SC)
# scpos = SC.index(scmax)
# kmax = K[scpos]

# ax1.plot(kmax, scmax, 'rx-')
# ax1.plot(kmin, dbimin, 'rx-')
# ax1.set_xlabel('Number of Clusters K')
# ax1.set_ylabel('Values of SC or DBI')
# ax1.set_title("Three Metrics to Determine the Optimal K for Velocity Data in Hierarchical Clustering",  fontdict={'fontsize': 10})
# # ax1.annotate('local max', xy=(kmax, scmax), xytext=(kmax, scmax+0.05), arrowprops=dict(facecolor='black', shrink=0.01),
# #             )
#
# ax2 = ax1.twinx()  # this is the important function
# ax2.plot(K, CHI, 'cx-', label = 'CHI')
# chimax = max(CHI)
# chipos = CHI.index(chimax)
# kmax0 = K[chipos]
# ax2.plot(kmax0, chimax, 'rx-')
# ax2.set_ylabel('Values of CHI')
# ax2.legend(loc=3, fontsize=10)
# plt.grid()
# plt.show()

# Sum_of_squared_distances = np.array(Sum_of_squared_distances)
# plt.figure(1)
# hcluster.dendrogram(linkage, leaf_font_size=10)
# p = hcluster.fcluster(linkage, n_clusters, criterion='maxclust')
# plt.show()

# X_test_pca, _, _ = pca(X_test_set, 2)
#
# plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=p, cmap='rainbow')


plt.plot(K, Sum_of_squared_distances, 'bx-')
# plt.semilogy(K, Sum_of_squared_distances)
plt.xlabel('Number of Clusters K')
plt.ylabel('Inertia')
plt.ylim([0, 3500])
# plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
plt.title('Elbow Method For Velocity Data in Hierarchical Clustering', fontsize = 11)
plt.grid()
plt.show()
