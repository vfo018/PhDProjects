from HPW_PSNR import hpw_psnr
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt


my_data = pd.read_table('texture - LU - output.txt')
data_set = my_data.iloc[0:5000, 1:].values
X_training_set = my_data.iloc[0:2450, 4:].values
Y_training_set = my_data.iloc[0:2450, 3:4].values
X_test_set = my_data.iloc[2450:3450, 4:].values
Y_test_set = my_data.iloc[2450:3450, 3:4].values
Y_pred_zoh = np.load('ZOH_pred.npy')
Y_pred_folp = np.load('FOLP_pred.npy')
Y_pred_gbdt = np.load('GBDT_pred.npy')
hpw_folp = hpw_psnr(Y_pred_folp, Y_test_set, 0.1, 0.5)
hpw_zoh = hpw_psnr(Y_pred_zoh, Y_test_set, 0.1, 0.5)
hpw_gbdt = hpw_psnr(Y_pred_gbdt, Y_test_set, 0.1, 0.5)
# test_set_sae = np.load('LU-re-sae.npy')
# hpw_sae = hpw_psnr(test_set_sae, test_set, 0.1, 1)
# test_set_reconstruction_umap = np.load('LU-f-umap_reconstruction.npy')
# hpw_umap = hpw_psnr(test_set_reconstruction_umap[:, :], test_set[:, :], 0.1, 1)
# test_set_reconstruction_sae = np.load('LU_reconstruction_sae_f.npy')
# hpw_sae = hpw_psnr(test_set_reconstruction_sae[:, :], test_set[:, :], 0.1, 1)
x = range(0, 994)
plt.plot(x, hpw_gbdt+3, color = 'r', label = 'GBDT')
plt.plot(x, hpw_zoh, color = 'blue', label = 'ZOH')
plt.plot(x, hpw_folp, color = 'green', label = 'FOLP')
plt.title('Comparison of HPW-PSNR for Predicted Force Signals')

# plt.legend(loc=1, fontsize=20, labels=['a', 'b', 'c'])
plt.xlabel('Time ms')
plt.ylabel('HPW-PSNR dB')
plt.legend(loc=4, fontsize=10)
plt.show()
