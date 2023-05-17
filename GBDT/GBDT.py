import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Data import Data
from pandas.core.frame import DataFrame


data = pd.read_table("../Data/texture - LD.dat")
# x_columns = []
# # for x in data.columns:
# #     if x in ['MV_x', 'MV_y', 'MV_z', 'MP_x', 'MP_y', 'MP_z']:
# #         x_columns.append(x)
# # X = data[x_columns]
# # y_columns = []
# # for y in data.columns:
# #     if y in ['SF_x', 'SF_y', 'SF_z']:
# #         y_columns.append(y)
# # Y = data[y_columns]
# # X_x = (np.array(X))[:, 0]
# # Y_x = (np.array(Y))[:, 0]
# # print(a[:, 0])
# # print(np.array(X))
# # Train along x axis
# data_x_columns = []
# for x1 in data.columns:
#     if x1 in ['MV_x', 'MP_x']:
#         data_x_columns.append(x1)
# # print(Y)
# X_x = data[data_x_columns]
# data_x_columns1 = []
# for y1 in data.columns:
#     if y1 in ['SF_x']:
#         data_x_columns1.append(y1)
# Y_x = data[data_x_columns1]
# # print(X_x)
# # print(Y_x)
# # X_train, X_test, Y_train, Y_test = train_test_split(X_x, Y_x, train_size=0.9, test_size=0.1, random_state=1)
# # print(X_train)
# X_train, X_test, Y_train, Y_test = X_x[0:22250], X_x[22250:], Y_x[0:22250], Y_x[22250:]
# # print(X_train)
# # print(Y_test)
# params = {'n_estimators': 500, 'max_depth': 7,
#           'learning_rate': 0.01, 'loss': 'huber'}
# gbr = ensemble.GradientBoostingRegressor(**params)
# gbr.fit(np.array(X_train), np.array(Y_train).ravel())
# Y_pred = gbr.predict(X_test)
# print(Y_pred)
# print(mean_squared_error(Y_test, Y_pred.ravel()))
# plt.figure(1)
# x = range(0, len(Y_test))
# plt.plot(x, Y_pred, 'g--')
# plt.plot(x, Y_test, 'b')
# plt.show()

# plt.figure(2)
# x = range(0, len(Y_test))
# plt.plot(x, Y_test, 'b')
# plt.show()
#
# plt.figure(3)
# x = range(0, len(Y_test))
# plt.plot(x, Y_pred, 'g--')
# plt.show()
# data_y_columns = []
# for x2 in data.columns:
#     if x2 in ['MV_y', 'MP_y']:
#         data_y_columns.append(x2)
# X_y = data[data_y_columns]
# data_y_columns1 = []
# for y2 in data.columns:
#     if y2 in ['SF_y']:
#         data_y_columns1.append(y2)
# Y_y = data[data_y_columns1]
# X_train1, X_test1, Y_train1, Y_test1 = X_y[0:22250], X_y[22250:], Y_y[0:22250], Y_y[22250:]
# params = {'n_estimators': 10, 'max_depth': None,
#           'learning_rate': 0.1, 'loss': 'ls'}
# gbr = ensemble.GradientBoostingRegressor(**params)
# gbr.fit(np.array(X_train1), np.array(Y_train1).ravel())
# Y_pred = gbr.predict(X_test1)
# print(Y_pred)
# print(mean_squared_error(Y_test1, Y_pred.ravel()))
# plt.figure(2)
# x = range(0, len(Y_test1))
# plt.plot(x, Y_pred, 'g--')
# plt.plot(x, Y_test1, 'b')
# plt.show()

# Nearly Perfect Prediction along z-axis
data_z_columns = []
my_data = Data('../Data/texture - LD.dat')

# for x3 in data.columns:
#     if x3 in ['MV_z', 'MP_z']:
#         data_z_columns.append(x3)
# X_z = data[data_z_columns]
# data_z_columns1 = []
# for y3 in data.columns:
#     if y3 in ['SF_z']:
#         data_z_columns1.append(y3)
# Y_z = data[data_z_columns1]
p_z = my_data.get_p('z')
v_z = my_data.get_v('z')
f_z = my_data.get_f('z')
dic = {"MP_z": p_z, "MV_z": v_z}
dic0 = {"SF_z": f_z}
X_z = DataFrame(dic)
Y_z = DataFrame(dic0)
# print(X_z)
# print(Y_z)

X_train2, X_test2, Y_train2, Y_test2 = X_z[0:7000], X_z[7000:], Y_z[0:7000], Y_z[7000:]
params = {'n_estimators': 100, 'max_depth': None,
          'learning_rate': 0.05, 'loss': 'ls'}
gbr = ensemble.GradientBoostingRegressor(**params)
gbr.fit(np.array(X_train2), np.array(Y_train2).ravel())
Y_pred = gbr.predict(X_test2)
print(Y_pred)
print(mean_squared_error(Y_test2, Y_pred.ravel()))

a = 0
Y_test_array = np.array(Y_test2)
for j in range(0, len(Y_test2)):
    d = np.sqrt((Y_test_array[j] - Y_pred[j]) ** 2)
    if Y_test_array[j] == 0 and Y_pred[j] <= 0.1:
        a += 1
    elif d <= 0.2 * abs(Y_test_array[j]):
        a += 1
per = a/len(Y_test_array)
print(per)

plt.figure(2)
x = range(0, len(Y_test2))
plt.plot(x, Y_pred, 'g--')
plt.plot(x, Y_test2, 'r')
plt.show()
