import numpy as np
from Parameter_Identification import Pi


def hc_update_top(data_f_all, data_p, data_v, data_f, index_z, threshold):
    p_train_all, v_train_all, f_train_all = [], data_v[:][:], []
    k_top, b_top, n_top = [], [], []
    p, v, f = [], [], []
    for i, q in enumerate(data_f_all):
        k_top.append(0.0)
        b_top.append(0)
        n_top.append(0)
        p.append(0)
        v.append(0)
        f.append(0)
    for i in range(0, len(data_p)):
        a = []
        for j in range(1, len(data_p[i])):
            a.append(data_p[i][j] - data_p[i][0])
        p_train_all.append(a)
    p_train_all = sum(p_train_all, [])
    # print(len(p_train_all))
    for j in range(0, len(data_v)):
        del v_train_all[j][0]
    v_train_all = sum(v_train_all, [])
    # print(len(v_train_all))
    index1 = []
    for k in range(0, len(data_f)):
        if data_f[k] == 0:
            index1.append(k)
    for l in range(0, len(index1) - 1):
        f_train_all.append(data_f[index1[l] + 1:index1[l + 1]])
    f_train_all = sum(f_train_all, [])
    # print(len(f_train_all))
    k_transmission_top, b_transmission_top, n_transmission_top = [], [], []
    my_PI = Pi(p_train_all, v_train_all, f_train_all)
    pm_initial = np.mat([[np.log(2000)], [10], [20]])
    p_initial = np.mat(1000 * np.eye(3))
    k_top1, b_top1, n_top1 = my_PI.pi_rls_rec(pm_initial, p_initial)
    # print(k_top1)
    # print(b_top1)
    a = 0
    for i in range(0, len(index_z), 2):
        k_top[index_z[i] + 1:index_z[i + 1] + 1] = k_top1[a: a + index_z[i + 1] - index_z[i]]
        b_top[index_z[i] + 1:index_z[i + 1] + 1] = b_top1[a: a + index_z[i + 1] - index_z[i]]
        n_top[index_z[i] + 1:index_z[i + 1] + 1] = n_top1[a: a + index_z[i + 1] - index_z[i]]
        p[index_z[i] + 1:index_z[i + 1] + 1] = p_train_all[a: a + index_z[i + 1] - index_z[i]]
        v[index_z[i] + 1:index_z[i + 1] + 1] = v_train_all[a: a + index_z[i + 1] - index_z[i]]
        f[index_z[i] + 1:index_z[i + 1] + 1] = f_train_all[a: a + index_z[i + 1] - index_z[i]]
        a += index_z[i + 1] - index_z[i]
    # print(k_top)
    # print(len(k_top))
    # print(f)
    # print(p)
    # print(v)
    transmission_ID = []
    num = 0
    for i in range(1, len(k_top)):
        if (k_top[i - 1] == 0 and k_top[i] != 0) or (k_top[i - 1] != 0 and k_top[i] == 0):
            # print(i)
            k_transmission_top.append(0)
            b_transmission_top.append(0)
            n_transmission_top.append(0)
            transmission_ID.append(i + 1)
            num += 1
        elif abs(f[i] - (k_top[i] * pow(abs(p[i]), n_top[i]) + b_top[i] * pow(abs(p[i]), n_top[i]) * abs(v[i]))) > \
                threshold * abs(f[i]):
            # print(i)
            k_transmission_top.append(k_top[i])
            b_transmission_top.append(b_top[i])
            n_transmission_top.append(n_top[i])
            transmission_ID.append(i + 1)
            num += 1
    # print(len(k_transmission_top))
    #  k, b, n update
    # if transmission_ID[-1] != len(k_top):
    #     transmission_ID.append(len(k_top))
    return k_transmission_top, b_transmission_top, n_transmission_top, transmission_ID, num, k_top1, b_top1, n_top1
