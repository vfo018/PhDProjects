from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from pandas.core.frame import DataFrame
import numpy as np


def update_op(data_p, data_v, data_f, train_num, transmission_ID):
    #     data_p, data_v refers to the original data in op side
    #     data_f refers to the receiving force in the op side
    data_p_train = data_p[0:train_num]
    data_v_train = data_v[0:train_num]
    data_f_train = data_f[0:train_num]
    data_p_test = data_p[train_num:]
    data_v_test = data_v[train_num:]

    #     Initialize model
    params = {'n_estimators': 100, 'max_depth': None,
              'learning_rate': 0.05, 'loss': 'ls'}
    dic = {"MP_z": data_p_train, "MV_z": data_v_train}
    dic0 = {"SF_z": data_f_train}
    X_z_train = DataFrame(dic)
    Y_z_train = DataFrame(dic0)
    gbr = ensemble.GradientBoostingRegressor(**params)
    gbr.fit(np.array(X_z_train), np.array(Y_z_train).ravel())

    #    Model Update
    dic1 = {"MP_z": data_p_test, "MV_z": data_v_test}
    X_z_test = DataFrame(dic1)
    data_f_output_op = data_f[0:train_num]
    # num = train_num
    a = 0
    # print(X_z_test[0])
    # Y_pred = gbr.predict(X_z_test)
    # # print(Y_pred)
    for i in range(0, len(data_p_test)):
        if i + 1 <= len(data_p_test):
            Y_pred = gbr.predict(X_z_test[i:i + 1])
        else:
            Y_pred = gbr.predict(X_z_test[i:])
        # print(Y_pred)
        if i in transmission_ID:
            data_f_output_op.append(data_f[train_num + a])
            #         Model Updates
            # print("Model update")
            data_p_train.append(data_p_test[i])
            data_v_train.append(data_v_test[i])
            data_f_train.append(data_f[train_num + a])
            dic = {"MP_z": data_p_train, "MV_z": data_v_train}
            dic0 = {"SF_z": data_f_train}
            X_z_train = DataFrame(dic)
            Y_z_train = DataFrame(dic0)
            gbr = ensemble.GradientBoostingRegressor(**params)
            gbr.fit(np.array(X_z_train), np.array(Y_z_train).ravel())
            a += 1
        else:
            data_f_output_op.extend(Y_pred.tolist())
        # per = len(data_p_test)
    return data_f_output_op
