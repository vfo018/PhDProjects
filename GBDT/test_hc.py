import numpy as np
import matplotlib.pyplot as plt
from Data import Data
from sklearn.metrics import mean_squared_error
from hcmodel_update_top import hc_update_top
from hcmodel_update_op import hc_update_op
from Parameter_Identification import Pi


# my_data = Data('newdata.txt')
# my_data = Data('Dynamic_iteraction_pressing_top_face_cube_25s_RecordedTOPSession_DB(1).txt')
my_data = Data('Static_interaction_tapping_surface_25s_RecordedTOPSession_DB.txt')
v_z = my_data.get_v('z')
p_z = my_data.get_p('z')
v_z_transmission, p_z_transmission, transmission_ID, num = my_data.data_transmission_op('z', 0.1)
f_z = my_data.get_f('z')
print(num)
v_z_top, p_z_top = my_data.data_receive_top('z', 0.05)
# print(v_z_top)
# print(p_z_top)
# print(len(v_z_top))
f_z_reduction, index_z = my_data.hc_zero_reduction(f_z)
# print(index_z)
v_z_top_reduction = my_data.hc_new_data(v_z_top, index_z)
p_z_top_reduction = my_data.hc_new_data(p_z_top, index_z)
# print(sum(v_z_top_reduction, []))
k_transmission_top, b_transmission_top, n_transmission_top, transmission_ID_top, num0, k_top, b_top, n_top = hc_update_top(f_z, p_z_top_reduction,
                                                                                            v_z_top_reduction,
                                                                                            f_z_reduction, index_z, 0.05)
print(num0)
k_receive = my_data.add_noise_f(k_transmission_top, 0, 0)
# print(k_receive)
b_receive = my_data.add_noise_f(b_transmission_top, 0, 0)
n_receive = my_data.add_noise_f(n_transmission_top, 0, 0)
l_num = len(v_z)
v_z_op_reduction = my_data.hc_new_data(v_z, index_z)
p_z_op_reduction = my_data.hc_new_data(p_z, index_z)
p_z_op, v_z_op = my_data.hc_get_p_v_op(p_z_op_reduction, v_z_op_reduction)
# print(p_z_op)
f_z_output = hc_update_op(k_receive, b_receive, n_receive, p_z_op, v_z_op, index_z, transmission_ID_top, l_num)
# print(f_z_output)
MSE = mean_squared_error(f_z, f_z_output)
print(MSE)
m = 0
# for i in range(0, len(f_z)):
#     if f_z_output[i] <= 0.1 and f_z[i] <= 0.1:
#         m += 1
#     elif abs(f_z[i] - f_z_output[i]) <= 0.1 * abs(f_z[i]):
#         m += 1
# per = m/len(f_z)
# print(per)
x = range(0, 2000)
# x = range(1500, 1600)
plt.plot(x, k_top[0:2000], 'r', label='k,  stiffness coefficient')
plt.plot(x, b_top[0:2000], 'g', label='b, damping coefficient')
plt.plot(x, n_top[0:2000], 'b', label='n, constant')
plt.legend()
plt.xlabel('Training Data')
plt.ylabel('Identification Value')
plt.axis([0, 2000, -10, 100])
plt.show()
# print(k_receive)
# print(k_op)
# print(len(k_op))
# print(len(v_z))
# print(k)
# print(b)
# print(n)
# print(len(k))
