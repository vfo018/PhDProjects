
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd





# label = cal_label(training_set)
# CON = my_calculator.continuity(50)
# TRU = my_calculator.trustworthiness(50)
# DPC = my_calculator.distancepearsoncorrelation()
# label = create_label(test_set, 0.1)


x = [0, 1, 2, 3, 4, 5]
DeadbandParameter = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]


# x_simple = np.array([-2, -1, 0, 1, 2])
# y_simple = np.array([4, 1, 3, 2, 0])
# DPC = distancepearsoncorrelation(x_simple, y_simple)

# for m in threshold:
#     bit = cal_transmission_bit(test_set_f, m, 'codecs')
#     total_bit_codecs_f.append(bit)
total_updates_ZOH = [568, 402, 271, 215, 175, 132]

total_updates_FOLP = [317, 223, 171, 139, 127, 104]
total_updates_GBDT = [259, 124, 77, 75, 70, 65]
# total_bit_umap_f = cal_transmission_bit(test_set_pca_f, 0.1, 'umap')
# total_bit_sae_f = cal_transmission_bit(test_set_pca_f, 0.1, 'sae')
# total_bit_pca_f = cal_transmission_bit(test_set_pca_f, 0.1, 'pca')
plt.figure(2)
plt.plot(x, total_updates_GBDT, color = 'blue', marker='o', label = 'GBDT')
plt.plot(x, total_updates_FOLP, color = 'green', marker='o', label = 'FOLP')
plt.plot(x, total_updates_ZOH, color = 'red', marker='o', label = 'ZOH')
plt.xticks(x, (0.02, 0.05, 0.1, 0.15, 0.2, 0.3))
# plt.axis([-0.02, 0.3, 0, 1])
plt.title('Comparison of Model Updates for Predicted Force Signals')
plt.xlabel('Deadband Parameter')
plt.ylabel('Number of Model Updates')
for a, b in zip(x, total_updates_GBDT):
    plt.text(a, b - 10, b, ha='center', va='top', fontsize=10)
for a, b in zip(x[:-1], total_updates_FOLP[:-1]):
    plt.text(a, b + 30, b, ha='center', va='top', fontsize=10)
plt.text(x[-1], total_updates_FOLP[-1]-10, total_updates_FOLP[-1], ha='center', va='top', fontsize=10)
for a, b in zip(x, total_updates_ZOH):
    plt.text(a, b + 10, b, ha='center', va='bottom', fontsize=10)

plt.legend()
plt.grid()
plt.show()
