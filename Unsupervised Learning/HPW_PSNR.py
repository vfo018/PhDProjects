import numpy as np


def hpw_psnr(data1, data2, threshold, k):
    diff = np.subtract(data1, data2)
    summation = 0
    data1_norm = []
    for i, v in enumerate(diff):
        summation += np.linalg.norm(v)**2
        data1_norm.append(np.linalg.norm(data1[i])**2)
    MSE = summation/len(diff)
    # print(MSE)
    max_index = data1_norm.index(max(data1_norm))
    min_index = data1_norm.index(min(data1_norm))
    diff_max = np.linalg.norm(np.subtract(data1[max_index], data1[min_index]))
    # print(diff_max)
    HPW = []
    for i, v in enumerate(data1):
        dis_vector = np.subtract(v, data2[i])
        dis = np.sqrt(sum(pow(dis_vector, 2)))
        if dis <= threshold * np.linalg.norm(v):
            c = 1
            HPW.append(c)
        else:
            # print(dis)
            # print(threshold * np.linalg.norm(v))
            c = k * (dis - threshold * np.linalg.norm(v)) + 1
            HPW.append(c)
    # print(HPW)
    HPW_PSNR = []
    for i, v in enumerate(HPW):
        a = 10 * np.log10(diff_max ** 2 / (MSE * v))
        HPW_PSNR.append(a)
    return np.array(HPW_PSNR)

# A = [[1, 2, 3], [3, 4, 4], [5, 6, 7]]
# B = [[1, 2, 3], [3, 4, 5], [2, 3, 4]]
# HPW_PSNR = hpw_psnr(A, B, 0.1, 1)


