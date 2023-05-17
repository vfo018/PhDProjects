import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PI import PI
from Data import Data
from sklearn.metrics import mean_squared_error
# from Criteria import Criteria


# This part is used to acquire data


# my_data = Data('Static_interaction_tapping_surface_25s_RecordedTOPSession_DB(1).txt')
my_data = Data('../Data/texture - LD.dat')
f_x1, index_x = my_data.zero_reduction('x')
f_y1, index_y = my_data.zero_reduction('y')
f_z1, index_z = my_data.zero_reduction('z')
# f_x = my_data.get_f('x')
# print(f_x1)
p_z1 = my_data.get_p('z')
new_p_z = my_data.new_data(p_z1, 'z')
v_z1 = my_data.get_v('z')
v_z = my_data.new_data(v_z1, 'z')
p_z = []
for i in range(0, len(new_p_z)):
    a = []
    for j in range(1, len(new_p_z[i])):
        a.append(new_p_z[i][j] - new_p_z[i][0])
    p_z.append(a)
# print(p_z[-1])
for j in range(0, len(v_z)):
    del v_z[j][0]
# print(v_z[-1])
index1 = []
for k in range(0, len(f_z1)):
    if f_z1[k] == 0:
        index1.append(k)
# print(index1)
f_z = []
for l in range(0, len(index1) - 1):
    f_z.append(f_z1[index1[l] + 1:index1[l+1]])
# f_z.pop()
# print(f_z[-1])
#     From here, data processing starts. HC-Model is implemented to derive a proper set of k, b and n
num = 25
# num0 = len(f_x) - num

# num represents the group number to be used in training data
f_z_training = sum(f_z[0:num][:], [])
v_z_training = sum(v_z[0:num][:], [])
p_z_training = sum(p_z[0:num][:], [])
# print(f_x_training)
my_PI = PI(p_z_training, v_z_training, f_z_training)
reg, x = my_PI.hc_linear()
pm_initial = np.mat([[np.log(2000)], [10], [20]])
p_initial = np.mat(1000 * np.eye(3))
pm, p = my_PI.pi_rls_rec(pm_initial, p_initial)
k = float(np.exp(pm[0]))
b = float(k * pm[1])
n = float(pm[2])
print(k)
print(b)
print(n)
#     Do the prediction from here
f_z_pred_abs = []
f_z_data = sum(f_z[num:][:], [])
v_z_data = sum(v_z[num:][:], [])
p_z_data = sum(p_z[num:][:], [])
# print(len(v_x_data))
# print(len(p_x_data))
for i in range(0, len(v_z_data)):
    f_z_pred_abs.append(k * pow(abs(p_z_data[i]), n) + b * pow(abs(p_z_data[i]), n) * (abs(v_z_data[i])))

f_z_pred = f_z_pred_abs[:]
# print(f_z_pred_abs)
# for k in range(1, len(f_x_data)):
#     if f_x_data[k] < 0:
#         f_x_pred[k] = -f_x_pred[k]

f_z_data_abs = []
for k in range(0, len(f_z_data)):
    f_z_data_abs.append(abs(f_z_data[k]))
# print(f_z_data_abs)
print(mean_squared_error(f_z_data_abs, f_z_pred))
# print(f_x_pred)
a = 0
for j in range(0, len(f_z_data)):
    d = np.sqrt((f_z_data[j] - f_z_pred[j]) ** 2)
    if d <= 0.2 * abs(f_z_data[j]):
        a += 1
per = a/len(f_z_data)
print(per)
#
plt.figure(1)
x = range(0, len(f_z_data))
plt.plot(x, f_z_data, 'g--')
plt.plot(x, f_z_pred, 'r')
plt.show()
