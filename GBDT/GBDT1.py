import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Data import Data
from pandas.core.frame import DataFrame


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
