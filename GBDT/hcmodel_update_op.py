import numpy as np
from Parameter_Identification import Pi


def hc_update_op(k_receive, b_receive, n_receive, data_p, data_v, index_ID, transmission_ID, l_num):
    # l_num is used to append 0
    # print(k_receive)
    k_op, b_op, n_op = [], [], []
    transmission_ID.insert(0, 0)
    data_f_output = []
    a = 0
    # print(transmission_ID)
    # print(len(transmission_ID))
    for i in range(1, len(transmission_ID)):
        for j in range(transmission_ID[i - 1], transmission_ID[i]):
            # print(j)
            k_op.append(k_receive[a - 1])
            b_op.append(b_receive[a - 1])
            n_op.append(n_receive[a - 1])
        a += 1
    k, b, n = k_receive[-1], b_op[-1], n_op[-1]
    while len(k_op) != l_num:
        k_op.append(k)
        b_op.append(b)
        n_op.append(n)
    for i in range(0, len(k_op)):
        data_f_output.append(0.0)
    a = 0
    # print(k_op)
    # print(data_p)
    for i in range(0, len(data_f_output)):
        for j in range(0, len(index_ID) - 1, 2):
            k = range(index_ID[j] + 1, index_ID[j + 1] + 1)
            # print(k)
            if i in k:
                data_f_output[i] = k_op[i] * pow(abs(data_p[a]), n_op[i]) + b_op[i] * pow(abs(data_p[a]), n_op[i]) * \
                                   abs(data_v[a])
                a += 1

    return data_f_output
