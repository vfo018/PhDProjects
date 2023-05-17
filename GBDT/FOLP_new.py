import numpy as np
import pandas as pd
from cal_pred_rate import cal_pred_rate
import matplotlib.pyplot as plt


my_data = pd.read_table('texture - LU - output.txt')
data_set = my_data.iloc[0:5000, 1:].values
Y_test_set = my_data.iloc[2450:3450, 3:4].values
Y_pred = [Y_test_set[0][0], Y_test_set[1][0]]
data_transmitted = [Y_test_set[0][0], Y_test_set[1][0]]
data_transmitted_index = [0, 1]
threshold = 0.3
m = 0
# for i, v in enumerate(Y_test_set[2:][:]):
#     if np.abs(data_transmitted[0]-data_transmitted[1]) != 0:
#         k = (data_transmitted[1]-data_transmitted[0])/(data_transmitted_index[1]-data_transmitted_index[0])
#         y = k * (i + 2 - data_transmitted_index[1]) + data_transmitted[1]
#         if np.abs(y-v[0]) <= threshold * np.abs(v[0]):
#             Y_pred.append(y)
#         else:
#             data_transmitted[0] = data_transmitted[1]
#             data_transmitted[1] = v[0]
#             data_transmitted_index[0] = data_transmitted_index[1]
#             data_transmitted_index[1] = i + 2
#             # Y_pred.append(data_transmitted[1])
#             Y_pred.append(y)
#             m += 1
#     else:
#         y = data_transmitted[1]
#         if np.abs(y-v[0]) <= threshold * np.abs(v[0]):
#             Y_pred.append(y)
#         else:
#             data_transmitted[0] = data_transmitted[1]
#             data_transmitted[1] = v[0]
#             data_transmitted_index[0] = data_transmitted_index[1]
#             data_transmitted_index[1] = i + 2
#             # Y_pred.append(data_transmitted[1])
#             Y_pred.append(y)
#             m += 1
#
# Y_pred = np.array(Y_pred)
# r, n = cal_pred_rate(Y_test_set, Y_pred, 0.1)
# np.save('FOLP_pred.npy', Y_pred)
Y_pred = np.load('FOLP_pred.npy')
#
plt.figure(0)
x = range(0, len(Y_test_set))
plt.plot(x, Y_test_set, 'g--',)
plt.plot(x, Y_pred, 'r')
plt.title('Comparison of Real and FOLP-predicted Force Feedback')
plt.xlabel('Time ms')
plt.ylabel('Force N')
plt.show()

plt.figure(1)
x = range(150, 400)
plt.plot(x, Y_test_set[150:400], 'g--', label = 'Real Force Feedback Along z-axis')
plt.plot(x, Y_pred[150:400], 'r', label = 'Force Feedback Predicted by FOLP Along z-axis')
plt.title('Comparison of Real and FOLP-predicted Force Feedback')
plt.xlabel('Time ms')
plt.ylabel('Force N')
plt.legend(loc=9)
plt.show()
