import numpy as np
import matplotlib.pyplot as plt
from Data import Data
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from model_update_top import update_top
from model_update_op import update_op



my_data = Data('Static_interaction_dragging_surface_25s_RecordedTOPSession_DB(1).txt')
p_z = my_data.get_p('y')
v_z = my_data.get_v('y')
f_z = my_data.get_f('y')
v_z_transmission, p_z_transmission, transmission_ID, num = my_data.data_transmission_op('z', 0.1)
print(num)
v_z_top, p_z_top = my_data.data_receive_top('z', 0.1)
print(0)
v_z_op_extend, p_z_op_extend = my_data.data_transmission_op_extend('z', 0.1)
print(1)
f_z_transmission, transmission_ID1, num0 = update_top(p_z_top, v_z_top, f_z, 5000, 0.1)
print(2)
f_z_receive = my_data.add_noise_f(f_z_transmission, 0, pow(10, -6))
print(num0)
f_z_output = update_op(p_z_op_extend, v_z_op_extend, f_z_receive, 5000, transmission_ID1)
MSE = mean_squared_error(f_z, f_z_output)
m = 0
for i in range(0, len(f_z)):
    if f_z_output[i] <= 0.1 and f_z[i] <= 0.1:
        m += 1
    elif abs(f_z[i] - f_z_output[i]) <= 0.1 * abs(f_z[i]):
        m += 1
per = m/len(f_z)
print(per)
x = range(0, len(f_z))
plt.title('Result Analysis')
plt.plot(x, f_z, 'g--', label='Actual friction')
plt.plot(x, f_z_output, 'r', label='Predictive friction from GBDT model')
plt.legend()
plt.show()
