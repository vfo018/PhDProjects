import numpy as np


def pca(v_original, dim_new):
    # calculate the mean value of a column
    v_original_mean = v_original.mean(axis=0)
    # print(v_original_mean)
    v_centralised = v_original - v_original_mean
    # print(v_centralised)
    v_centralised_correlation = np.dot(v_centralised.T, v_centralised) * 1/v_centralised.shape[1]
    # print(v_centralised_correlation)
    eig_value, eig_vec = np.linalg.eig(v_centralised_correlation)
    # print(eig_vec)
    # print(eig_value)
    v_dim_reduction = np.dot(v_original, eig_vec[:, 0:dim_new])
    v_reconstruction = np.dot(v_dim_reduction, eig_vec[:, 0:dim_new].T)
    return v_dim_reduction, eig_vec, v_reconstruction

# A = np.array([[-1, -2], [-1, 0], [0, 0], [2, 1], [0, 1]])
# A = np.array([[0.03486833, 0.10165667, -0.330111], [0.00329933, 0.01021767, -0.033083],
#               [-0.03816767, -0.11187433, 0.363194]])

# A = np.array([[0.5, 1, 2], [1.1, 1.2, 1.3], [0.5, 0.15, 0.25]])
# a, b, c = pca(A, 2)
# b_inv = np.matrix(b[:, 0:2]).I
# c = np.dot(b[:, 0:2], a.T).T
# c0 = np.dot(a, b_inv)
# c1 = np.dot(a, b[:, 0:2].T)