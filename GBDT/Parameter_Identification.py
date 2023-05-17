import numpy as np


class Pi:
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
        k, b, n = [], [], []
        ff1 = 1
        for a in range(0, len(y0)):
            if a == 0:
                # 设置初始条件
                # p_pr1 = np.mat(1000 * np.eye(3))
                # pm_pr1 = np.mat([[np.log(2000)], [2], [2]])
                p_pr1 = p_initial
                pm_pr1 = pm_initial
                k0 = np.exp(pm_pr1[0]).tolist()
                k.append(k0)
                b.append((k0 * pm_pr1[1]).tolist())
                n.append(pm_pr1[2].tolist())
                # print(pm_pr1)
                # print(p_pr1)
            else:
                # print(p_pr1)
                x = reg[a]
                pm_pr2, p_pr2 = self.pi_rls(pm_pr1, p_pr1, x, ff1, y0[a])
                # print(pm_pr1)
                # print(p_pr1)
                pm_pr1, p_pr1 = pm_pr2, p_pr2
                k0 = np.exp(pm_pr1[0]).tolist()
                k.append(k0)
                b.append((k0 * pm_pr1[1]).tolist())
                n.append(pm_pr1[2].tolist())
                # print(pm_pr1)
                # print(p_pr1)
                # pm
                # pm_pr2, p_pr2 = 0, 0
        k = sum(sum(k, []), [])
        b = sum(sum(b, []), [])
        n = sum(sum(n, []), [])
        return k, b, n
