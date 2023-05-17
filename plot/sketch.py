import numpy as np
import matplotlib.pyplot as plt
from Data import Data
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from model_update_top import update_top
from model_update_op import update_op
from hcmodel_update_top import hc_update_top
from hcmodel_update_op import hc_update_op
from Parameter_Identification import Pi


my_data1 = Data('Dynamic_iteraction_pressing_top_face_cube_25s_RecordedTOPSession_DB(1).txt')
v_z1 = my_data1.get_v('z')
p_z1 = my_data1.get_p('z')
v_z_transmission, p_z_transmission, transmission_ID, num = my_data1.data_transmission_op('z', 0.1)
f_z1 = my_data1.get_f('z')
# print(num)
v_z_top, p_z_top = my_data1.data_receive_top('z', 0.05)
f_z_reduction, index_z = my_data1.hc_zero_reduction(f_z1)
v_z_top_reduction = my_data1.hc_new_data(v_z_top, index_z)
p_z_top_reduction = my_data1.hc_new_data(p_z_top, index_z)
k_transmission_top, b_transmission_top, n_transmission_top, transmission_ID_top, num0, f_top1 = hc_update_top(f_z1, p_z_top_reduction,
                                                                                            v_z_top_reduction,
                                                                                            f_z_reduction, index_z, 0.1)
# print(num0)
k_receive = my_data1.add_noise_f(k_transmission_top, 0, 0)
b_receive = my_data1.add_noise_f(b_transmission_top, 0, 0)
n_receive = my_data1.add_noise_f(n_transmission_top, 0, 0)
l_num = len(v_z1)
v_z_op_reduction = my_data1.hc_new_data(v_z1, index_z)
p_z_op_reduction = my_data1.hc_new_data(p_z1, index_z)
p_z_op, v_z_op = my_data1.hc_get_p_v_op(p_z_op_reduction, v_z_op_reduction)
f_z_output1 = hc_update_op(k_receive, b_receive, n_receive, p_z_op, v_z_op, index_z, transmission_ID_top, l_num)
# MSE = mean_squared_error(f_z, f_z_output)
# print(MSE)
# for i in range(0, len(f_z)):
#     if f_z_output[i] <= 0.1 and f_z[i] <= 0.1:
#         m += 1
#     elif abs(f_z[i] - f_z_output[i]) <= 0.1 * abs(f_z[i]):
#         m += 1
# per = m/len(f_z)
# print(per)
my_data = Data('Dynamic_iteraction_pressing_top_face_cube_25s_RecordedTOPSession_DB(1).txt')
# my_data = Data('newdata.txt')
p_z = my_data.get_p('z')
v_z = my_data.get_v('z')
f_z = my_data.get_f('z')
# v_z_transmission, p_z_transmission, transmission_ID, num = my_data.data_transmission_op('z', 0.1)
# print(num)

v_z_top, p_z_top = my_data.data_receive_top('z', 0.1)

f_z_transmission, transmission_ID, num0 = update_top(p_z_top, v_z_top, f_z, 1000, 0.1)
f_z_receive = my_data.add_noise_f(f_z_transmission, 0, pow(10, -6))
# print(num0)
v_z_op_extend, p_z_op_extend = my_data.data_transmission_op_extend('z', 0.1)
f_z_output = update_op(p_z_op_extend, v_z_op_extend, f_z_receive, 1000, transmission_ID)
# MSE = mean_squared_error(f_z, f_z_output)
# print(MSE)
m = 0
MSE = []
MSE1 =[]
# v_z_op_extend, p_z_op_extend = my_data.data_transmission_op_extend('z', 0.1)
for i in range(500, 1550, 50):
    print(i)
    f_z_transmission, transmission_ID, num0 = update_top(p_z_top, v_z_top, f_z, i, 0.1)
    f_z_receive = my_data.add_noise_f(f_z_transmission, 0, 0)
    # print(num0)
    # v_z_op_extend, p_z_op_extend = my_data.data_transmission_op_extend('z', 0.1)
# f_z_output = update_op(p_z, v_z, f_z_receive, 2000, transmission_ID)
    f_z_output = update_op(p_z_op_extend, v_z_op_extend, f_z_receive, i, transmission_ID)
    f_z_output1 = hc_update_op(k_receive, b_receive, n_receive, p_z_op, v_z_op, index_z, transmission_ID_top, l_num)
    MSE.append(mean_squared_error(f_z, f_z_output))
    MSE1.append(mean_squared_error(f_z, f_top1))
print(MSE)
x = range(500, 1550)
plt.title('Result Analysis')

plt.plot(x, MSE, 'r', lable='training data')
# plt.plot(x, f_z_output, 'r')
plt.legend()
plt.xlabel('Training Data')
plt.ylabel('MSE')
plt.show()
x = range(0, len(f_z))
plt.plot(x, f_z, 'g--')
plt.plot(x, f_z_output1, 'r')
plt.plot(x, f_z_output, 'b')
plt.show()
