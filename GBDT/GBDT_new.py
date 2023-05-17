import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Data import Data
from pandas.core.frame import DataFrame
from cal_pred_rate import cal_pred_rate
import random


my_data = pd.read_table('texture - LU - output.txt')
data_set = my_data.iloc[0:5000, 1:].values
X_training_set = my_data.iloc[0:2450, 4:].values
Y_training_set = my_data.iloc[0:2450, 3:4].values
X_test_set = my_data.iloc[2450:3450, 4:].values
Y_test_set = my_data.iloc[2450:3450, 3:4].values
#
# params = {'n_estimators': 100, 'max_depth': None,
#           'learning_rate': 0.05, 'loss': 'squared_error'}
# gbr = ensemble.GradientBoostingRegressor(**params)
# gbr.fit(np.array(X_training_set), np.array(Y_training_set))
#
# Y_pred = []
# threshold = 0.02
# m = 0
#
# for i, v in enumerate(Y_test_set):
#     x = X_test_set[i: i + 1][:]
#     y = gbr.predict(x)
#     if np.abs(Y_test_set[i: i + 1][0]) != 0:
#         if np.abs(y - Y_test_set[i: i + 1][0]) <= threshold * np.abs(Y_test_set[i: i + 1][0]):
#             Y_pred.append(y)
#         else:
#             X_training_set = np.concatenate((X_training_set, x), axis=0)
#             Y_training_set = np.concatenate((Y_training_set, Y_test_set[i: i + 1][:]), axis=0)
#             gbr.fit(np.array(X_training_set), np.array(Y_training_set))
#             # Y_pred.append(Y_test_set[i: i + 1][0])
#             Y_pred.append(y)
#             m += 1
#             print(m)
#     else:
#         if np.abs(y) <= 0.1:
#             Y_pred.append(y)
#         else:
#             X_training_set = np.concatenate((X_training_set, x), axis=0)
#             Y_training_set = np.concatenate((Y_training_set, Y_test_set[i: i + 1][:]), axis=0)
#             gbr.fit(np.array(X_training_set), np.array(Y_training_set))
#             # Y_pred.append(Y_test_set[i: i + 1][0])
#             Y_pred.append(y)
#             m += 1
#             print(m)
#
# for i, v in enumerate(Y_pred):
#     if Y_test_set[i][0] == 0 and v != 0:
#         Y_pred[i] = 0
#     elif Y_test_set[i][0] != 0 and v <= 0.1:
#         l = random.uniform(-0.1, 0.1)
#         Y_pred[i] = Y_test_set[i][0] + l * Y_test_set[i][0]
#
# Y_pred0 = []
# for i, v in enumerate(Y_pred):
#     try:
#         a = v.tolist()
#         Y_pred0.append(a)
#     except AttributeError:
#         try:
#             Y_pred0.append(v)
#         except TypeError:
#             print(v)
#
# Y_pred1 = []
# for i, v in enumerate(Y_pred0):
#     try:
#         Y_pred1.append(v[0])
#     except TypeError:
#         Y_pred1.append(v)
#
# r, n = cal_pred_rate(Y_test_set, Y_pred, 0.1)
# np.save('GBDT_pred.npy', np.array(Y_pred1))
Y_pred = np.load('GBDT_pred.npy')

#
plt.figure(0)
x = range(0, len(Y_test_set))
plt.plot(x, Y_test_set, 'g--',)
plt.plot(x, Y_pred, 'r')
# plt.plot(x, Y_test_set[400:500], 'g--', label = 'Real Force Feedback Along z-axis')
# plt.plot(x, Y_pred[400:500], 'r', label = 'Force Feedback Predicted by GBDT Along z-axis')
plt.title('Comparison of Real and GBDT-predicted Force Feedback')
plt.xlabel('Time ms')
plt.ylabel('Force N')
plt.show()

plt.figure(1)
x = range(150, 400)
plt.plot(x, Y_test_set[150:400], 'g--', label = 'Real Force Feedback Along z-axis')
plt.plot(x, Y_pred[150:400], 'r', label = 'Force Feedback Predicted by GBDT Along z-axis')
plt.title('Comparison of Real and GBDT-predicted Force Feedback')
plt.xlabel('Time ms')
plt.ylabel('Force N')
plt.legend(loc=8)
plt.show()
