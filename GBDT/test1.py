from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from pandas.core.frame import DataFrame
import numpy as np
from Data import Data


my_data = Data('newdata.txt')
p_z = my_data.get_p('z')
v_z = my_data.get_v('z')
f_z = my_data.get_f('z')
v_z_transmission, p_z_transmission, transmission_ID, num = my_data.data_transmission_op('z', 0.1)
v_z_top, p_z_top = my_data.data_receive_top('z', 0.1)
# print(len(v_z_top))
# print(len(v_z))
# print(v_z_top)
# print(v_z)
print(v_z_transmission)
print(len(v_z_transmission))
print(len(v_z))
# data_p_train = p_z[0:2000]
# data_v_train = v_z[0:2000]
# data_f_train = f_z[0:2000]
# dic = {"MP_z": data_p_train, "MV_z": data_v_train}
# dic0 = {"SF_z": data_f_train}
# X_z_train = DataFrame(dic)
# Y_z_train = DataFrame(dic0)
# print(X_z_train[0:1])
