import numpy as np


class PI:
    def __init__(self, p, v, f):
        self.p = p
        self.v = v
        self.f = f

    def hc_linear(self):
        # Linearlize Hunt-Crossley Model for RLS
        # p represents position, v for velocity and f for force feedback
        reg = np.zeros((1, 3))
        y = []
        for i in range(0, len(self.p)):
            reg = np.row_stack((reg, [1, float(abs(self.v[i])), float(np.log(abs(self.p[i])))]))
            y.append(np.log(abs(self.f[i])))
        return np.delete(reg, 0, axis=0), y

    def pi_rls(self, pm_pr, p_pr, reg, ff, y):
        # pm_pr for Θ(t-1), p_pr for P(t-1), reg for ψ(t), ff for forgetting factor λ, y for y(t)
        # pm = np.mat([])
        p = 1 / ff * (np.mat(p_pr) - (np.mat(p_pr) * np.mat(reg).T * np.mat(reg) * np.mat(p_pr)) /
                      (ff + np.mat(reg) * np.mat(p_pr) * np.mat(reg).T))
        pm = np.mat(pm_pr) + np.mat(p) * np.mat(reg).T * (y - np.mat(reg) * np.mat(pm_pr))
        return pm, p

    def pi_rls_rec(self, pm_initial, p_initial):
        # pm_pr for Θ(t-1), p_pr for P(t-1), reg for ψ(t), ff for forgetting factor λ, y for y(t)
        # pm = np.mat([])
        reg, y0 = self.hc_linear()
        ff1 = 1
        for a in range(0, len(y0)):
            if a == 0:
                # 设置初始条件
                # p_pr1 = np.mat(1000 * np.eye(3))
                # pm_pr1 = np.mat([[np.log(2000)], [2], [2]])
                p_pr1 = p_initial
                pm_pr1 = pm_initial
                # print(pm_pr1)
                # print(p_pr1)
            else:
                # print(p_pr1)
                x = reg[a]
                pm_pr2, p_pr2 = self.pi_rls(pm_pr1, p_pr1, x, ff1, y0[a])
                # print(pm_pr1)
                # print(p_pr1)
                pm_pr1, p_pr1 = pm_pr2, p_pr2
                print(pm_pr1)
                print(p_pr1)
                # pm_pr2, p_pr2 = 0, 0
        return pm_pr1, p_pr1


# Testing Code
# Y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# p_pr1 = []
# pm_pr1 = []
# reg1 = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
#         [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11],
#         [10, 11, 12]]
# ff1 = 0.1
# for a in range(0, len(Y)):
#     if a == 0:
#         # 设置初始条件
#         p_pr1 = np.mat(np.ones((3, 3)))
#         pm_pr1 = np.mat(np.ones((3, 1)))
#     else:
#         # print(p_pr1)
#         x = reg1[a]
#         pm_pr2, p_pr2 = pi_rls(pm_pr1, p_pr1, x, ff1, Y[a])
#         print(pm_pr1)
#         print(p_pr1)
#         pm_pr1, p_pr1 = pm_pr2, p_pr2
#         print(pm_pr1)
#         print(p_pr1)
# print(np.mat([12, 13, 14]) * pm_pr2)



