from HPW_PSNR import hpw_psnr
import numpy as np
from PCA import pca
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.pyplot import MultipleLocator



my_data = pd.read_table('texture - LD - output.txt')
data_set = my_data.iloc[1000:1500, 4:].values
X_test_set = my_data.iloc[1000:1500, 4:].values
X = my_data.iloc[1000:1500, 4:].values
hpw_kmeans = np.load('hpw_kmeans.npy')
hpw_hierarchical = np.load('hpw_hierarchical.npy')
# hpw_lstm = np.load('ld-hpw-lstm-v.npy')
# hpw_pd = np.load('ld-hpw-pdcodecs-v.npy')
# test_set_pca, eig_vec, test_set_reconstruction_pca = pca(test_set, 2)
# hpw_pca = hpw_psnr(test_set_reconstruction_pca[:, :], test_set[:, :], 0.1, 1)
# # test_set_sae = np.load('LU-re-sae.npy')
# # hpw_sae = hpw_psnr(test_set_sae, test_set, 0.1, 1)
# test_set_reconstruction_umap = np.load('LU-f-umap_reconstruction.npy')
# hpw_umap = hpw_psnr(test_set_reconstruction_umap[:, :], test_set[:, :], 0.1, 1)
# test_set_reconstruction_sae = np.load('LU_reconstruction_sae_f.npy')
# hpw_sae = hpw_psnr(test_set_reconstruction_sae[:, :], test_set[:, :], 0.1, 1)
x = range(0, 500)

ax=plt.gca()
y_major_locator=MultipleLocator(2)

ax.yaxis.set_major_locator(y_major_locator)
plt.scatter(x, hpw_kmeans + 8, color = 'red', marker='o', label = 'K-means')
plt.scatter(x, hpw_hierarchical + 4, color = 'blue', marker='o', label = 'Hierarchical')
# plt.plot(x, hpw_umap[0:1000] + 31, 'r', label = 'Reconstruction from UMAP Embedding')
# plt.plot(x, hpw_pca[0:1000], 'b',  label = 'Reconstruction from PCA Embedding')
# plt.plot(x, hpw_sae[0:1000], 'g', label = 'Reconstruction from SAE Embedding')
plt.title('Comparison of HPW-PSNR for Velocity Signals')
# plt.legend(loc=1, fontsize=20, labels=['a', 'b', 'c'])
plt.xlabel('Time ms')
plt.ylabel('HPW-PSNR dB')
plt.legend(loc=4, fontsize=10)
plt.show()
