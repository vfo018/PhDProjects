import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PI import PI
from Data import Data
from sklearn.metrics import mean_squared_error
from Criteria import Criteria

my_data = Data('data 1.txt')
p_z = my_data.get_p('z')
v_z = my_data.get_v('z')
f_z = my_data.get_f('z')
# print(p_x)
my_PI = PI(p_z, v_z, f_z)
reg, y = my_PI.hc_linear()
pm_initial = np.mat([[np.log(2000)], [2], [2]])
p_initial = np.mat(1000 * np.eye(3))
pm, p = my_PI.pi_rls_rec(pm_initial, p_initial)
# print(pm)
k = np.exp(pm[0])
b = k * pm[1]
n = sum(pm[2].tolist(), [])
# print(k)
# print(b)
# print(n)
# print(p_z[-1], v_z[-1])
# f = k * pow(-p_z[-1], n[0]) + b * pow(-p_z[-1], n[0]) * (-v_z[-1])
# print(f)
my_data1 = Data('data 2.txt')
p_z1 = my_data1.get_p('z')
v_z1 = my_data1.get_v('z')
f_z1 = my_data1.get_f('z')
my_PI1 = PI(p_z1, v_z1, f_z1)
reg1, y1 = my_PI1.hc_linear()
pm1, p1 = my_PI.pi_rls_rec(pm, p)
k1 = float(np.exp(pm1[0]))
b1 = float(k * pm1[1])
n1 = float(pm1[2])
print(k1)
print(b1)
print(n1)
# f = k1 * pow(-p_z1[-1], n1[0]) + b1 * pow(-p_z1[-1], n1[0]) * (-v_z1[-1])
# print(f)
my_data2 = Data('data 3.txt')
p_z2 = my_data2.get_p('z')
v_z2 = my_data2.get_v('z')
f_z2 = my_data2.get_f('z')
f_z_pred = []
for i in range(0, len(p_z2)):
    f_z_pred.append(k1 * pow(-p_z2[i], n1) + b1 * pow(-p_z2[i], n1) * (-v_z2[i]))

print(mean_squared_error(f_z2, f_z_pred))
# print(f_z_pred)
a = 0
for j in range(0, len(f_z2)):
    d = np.sqrt((f_z2[j] - f_z_pred[j]) ** 2)
    if d <= 0.2 * abs(f_z2[j]):
        a += 1
per = a/len(f_z2)
print(per)

plt.figure(1)
x = range(0, len(f_z2))
plt.plot(x, f_z2, 'g--')
plt.plot(x, f_z_pred, 'b')
plt.show()
a = 0

