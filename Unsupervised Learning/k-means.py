import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from HPW_PSNR import hpw_psnr
from sklearn import metrics


my_data = pd.read_table('texture - LD - output.txt')
data_set = my_data.iloc[:, 1:4].values
X_test_set = my_data.iloc[1000:1500, 1:4].values
X = my_data.iloc[1000:1500, 1:4].values
# n_clusters = 7
#
# km = KMeans(n_clusters).fit(X_test_set)
# labels = km.predict(X_test_set)
# centroids = km.cluster_centers_
# samplesCentroids = centroids[labels]
# count_label = np.bincount(labels)
# sort_index = np.argsort(-count_label)
# The percentage of clusterings to be transmitted
# per = 2/3
# for i, v in enumerate(X):
#     if labels[i] in sort_index[int(n_clusters * per):]:
#         X[i] = samplesCentroids[i][:]
# hpw_kmeans_f = hpw_psnr(X_test_set, X, 0.1, 1)
# r_kmeans = max(count_label)/500
# np.save('hpw_kmeans_f', hpw_kmeans_f)
Sum_of_squared_distances = []
K = range(2, 31)
for k in K:
    km0 = KMeans(n_clusters=k)
    km0 = km0.fit(X_test_set)
    Sum_of_squared_distances.append(km0.inertia_)

#
# SC, CHI, DBI = [], [], []
# K = range(5, 16)
# for k in K:
#     km0 = KMeans(n_clusters=k)
#     km0 = km0.fit(X_test_set)
#     labels = km0.labels_
#     sc = metrics.silhouette_score(X_test_set, labels, metric='euclidean')
#     chi = metrics.calinski_harabasz_score(X_test_set, labels)
#     dbi = metrics.davies_bouldin_score(X_test_set, labels)
#     SC.append(sc)
#     CHI.append(chi)
#     DBI.append(dbi)

# #
# # x = range(5, 100, 1)
# #
# # plt.plot(K, Sum_of_squared_distances)
# # plt.scatter(K, Sum_of_squared_distances)
# #
# #
# #
# # plt.xticks(K)
# # plt.yticks([])  # 去掉y轴上的刻度显示
# # plt.xlabel("K", fontsize=13)
# # plt.ylabel("distance", fontsize=13)
# # plt.show()
#
#
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('Number of Clusters K')
plt.ylabel('Inertia')
plt.title('Elbow Method For Velocity Data in K-means Clustering', fontsize=11)
plt.ylim([0, 3500])
# plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
plt.grid()
plt.show()
# #
# fig = plt.figure()
# #
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
# ax1.set_title("Three Metrics to Determine the Optimal K for Velocity Data in K-means Clustering",  fontdict={'fontsize': 10})
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
# ax2.legend(loc=6, fontsize=10)
#
# # plt.title("Three Metrics to Determine the Optimal K for Force Data in K-means Clustering")
# # plt.plot(K, SC)
# # plt.xlabel('Number of Clusters K')
# # plt.ylabel('SC')
# # plt.xlabel('Number of Clusters K')
# # plt.ylabel('HPW-PSNR dB')
# # plt.legend(loc=5, fontsize=10)
# plt.grid()
# plt.show()


# x = range(0, len(X))
# # plt.plot(x, hpw_umap, color = 'red', marker='o', label = 'UMAP', s=1)
# # plt.plot(x, hpw_pca, color = 'blue', marker='o', label = 'PCA', s=1)
# # plt.plot(x, hpw_sae, color = 'green', marker='o', label = 'PCA', s=1)
# plt.plot(x, HPW, 'r', label = 'Reconstruction from UMAP Embedding')
# plt.title('Comparison of HPW-PSNR for Reconstructed Force Signals')
# # plt.legend(loc=1, fontsize=20, labels=['a', 'b', 'c'])
# plt.xlabel('Time ms')
# plt.ylabel('HPW-PSNR dB')
# plt.legend(loc=4, fontsize=10)
# plt.show()
