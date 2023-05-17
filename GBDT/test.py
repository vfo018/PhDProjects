import numpy as np
import matplotlib.pyplot as plt
from Data import Data
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from model_update_top import update_top
from model_update_op import update_op


# my_data = Data('Static_interaction_tapping_surface_25s_RecordedTOPSession_DB.txt')
my_data = Data('Dynamic_iteraction_pressing_top_face_cube_25s_RecordedTOPSession_DB(1).txt')
# my_data = Data('newdata.txt')
p_z = my_data.get_p('z')
v_z = my_data.get_v('z')
f_z = my_data.get_f('z')
v_z_transmission, p_z_transmission, transmission_ID, num = my_data.data_transmission_op('z', 0.1)
print(num)
# v_z_transmission_noise = my_data.add_noise('v', 'z', 0.1, 0, pow(10, -6))
# p_z_transmission_noise = my_data.add_noise('p', 'z', 0.1, 0, pow(10, -6))
v_z_top, p_z_top = my_data.data_receive_top('z', 0.1)
# print(v_z_transmission)
# print(v_z_transmission_noise)
MSE = []
v_z_op_extend, p_z_op_extend = my_data.data_transmission_op_extend('z', 0.1)
for i in range(100, 2050, 50):
    print(i)
    f_z_transmission, transmission_ID, num0 = update_top(p_z_top, v_z_top, f_z, i, 0.1)
    f_z_receive = my_data.add_noise_f(f_z_transmission, 0, 0)
    # print(num0)
    # v_z_op_extend, p_z_op_extend = my_data.data_transmission_op_extend('z', 0.1)
# f_z_output = update_op(p_z, v_z, f_z_receive, 2000, transmission_ID)
    f_z_output = update_op(p_z_op_extend, v_z_op_extend, f_z_receive, i, transmission_ID)
    MSE.append(mean_squared_error(f_z, f_z_output))
print(MSE)
x = range(100, 2050, 50)
plt.title('Result Analysis')

plt.plot(x, MSE, 'r', label='training data')
plt.axis([100, 2000, -10, 100])
# plt.plot(x, f_z_output, 'r')
plt.legend()
plt.xlabel('Training Data')
plt.ylabel('MSE')
plt.show()
# m = 0
# for i in range(0, len(f_z)):
#     if f_z_output[i] <= 0.1 and f_z[i] <= 0.1:
#         m += 1
#     elif abs(f_z[i] - f_z_output[i]) <= 0.1 * abs(f_z[i]):
#         m += 1
# per = m/len(f_z)
# print(per)
# x = range(0, len(f_z))
# plt.plot(x, f_z, 'g--')
# plt.plot(x, f_z_output, 'r')
# plt.show()

# print(len(v_z))
# print(v_z_transmission)
# print(len(v_z_transmission))
# print(p_z_transmission)
# print(len(p_z_transmission))
# print(p_z)
# print(v_z)
# print(f_z)
