from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from pandas.core.frame import DataFrame
import numpy as np


def update_top(data_p, data_v, data_f, train_num, threshold):
    #     data_p, data_v refers to the received data in the top side
    #     data_f refers to the original data in the op side
    data_p_train = data_p[0:train_num]
    data_v_train = data_v[0:train_num]
    data_f_train = data_f[0:train_num]
    data_p_test = data_p[train_num:]
    data_v_test = data_v[train_num:]
    data_f_test = data_f[train_num:]

    #     Initialize model
    params = {'n_estimators': 10, 'max_depth': 3,
              'learning_rate': 0.05, 'loss': 'ls'}
    dic = {"MP_z": data_p_train, "MV_z": data_v_train}
    dic0 = {"SF_z": data_f_train}
    X_z_train = DataFrame(dic)
    Y_z_train = DataFrame(dic0)
    gbr = ensemble.GradientBoostingRegressor(**params)
    gbr.fit(np.array(X_z_train), np.array(Y_z_train).ravel())
    print('a')

    #    Model Update
    dic1 = {"MP_z": data_p_test, "MV_z": data_v_test}
    X_z_test = DataFrame(dic1)
    data_f_transmission_top = data_f[0:train_num]
    data_f_not_transmission_top = []
    transmission_ID = []
    num = 0
    # print(X_z_test[0])
    # Y_pred = gbr.predict(X_z_test)
    # # print(Y_pred)
    for i in range(0, len(data_p_test)):
        if i + 1 <= len(data_p_test):
            Y_pred = gbr.predict(X_z_test[i:i + 1])
        else:
            # print(i)
            Y_pred = gbr.predict(X_z_test[i:])
        # print(Y_pred)
        if abs(Y_pred) >= 0.1 and abs(data_f_test[i]) >= 0.1:
            if abs(Y_pred - data_f_test[i]) >= threshold * abs(data_f_test[i]):
                data_f_transmission_top.append(data_f_test[i])
                num += 1
                transmission_ID.append(i)
                #         Model Updates
                print("Model Updates")
                data_p_train.append(data_p_test[i])
                data_v_train.append(data_v_test[i])
                data_f_train.append(data_f_test[i])
                dic = {"MP_z": data_p_train, "MV_z": data_v_train}
                dic0 = {"SF_z": data_f_train}
                X_z_train = DataFrame(dic)
                Y_z_train = DataFrame(dic0)
                gbr = ensemble.GradientBoostingRegressor(**params)
                gbr.fit(np.array(X_z_train), np.array(Y_z_train).ravel())
            else:
                data_f_not_transmission_top.extend(Y_pred.tolist())
                print("0000000000000000000000000000000000000000000000000")
                # data_f_transmission_top.extend(Y_pred.tolist())
        else:
            data_f_not_transmission_top.extend(Y_pred.tolist())
            # data_f_transmission_top.extend(Y_pred.tolist())
    return data_f_transmission_top, transmission_ID, num
