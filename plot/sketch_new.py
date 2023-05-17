import matplotlib.pyplot as plt
import numpy as np


A = [1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.1, 1.9, 2.2, 2.4]
C = [1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.1, 1.9, 2.2, 2.5]
B = [1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.0, 2.0, 2.0, 2.0, 2.4]
D = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [10, 10]]
D0 = [[0, 0], [1, 1]]
D1 = [[7, 7], [8, 8], [9, 9]]
E = [[0, 0.96], [0, 1.16], [0, 1.36], [0, 1.56], [0, 1.76], [0, 1.96], [0, 2.16], [0, 2.06], [0, 1.86], [0, 2.12], [0, 2.36]]
F = [[0, 0.96], [0, 1.16], [0, 1.36], [0, 1.56], [0, 1.76], [0, 1.96], [0, 2.36]]
F0 = [[0, 0.96], [0, 1.16]]
F1 = [[0, 2.06], [0, 1.86], [0, 2.16]]
J = [[0, 0.96], [0, 1.16], [0, 1.26], [0, 1.36], [0, 1.76], [0, 1.96], [0, 2.36]]
G = [[0, 2.16], [0, 2.06], [0, 1.86], [0, 2.16]]
G0 = [[0, 1.36], [0, 1.56], [0, 1.76], [0, 1.96], [0, 2.16]]
G1 = [[0, 2.46]]
I = [[0, 1.96], [0, 1.96], [0, 1.96], [0, 1.96]]
H = [[6, 6], [7, 7], [8, 8], [9, 9]]
H0 = [[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]
H1 = [[10, 10]]

x = range(0, len(B))

# plt.plot(x, B, 'r', label='Predictive suppression from GBDT model')
# plt.show()
ax = plt.gca()
# plt.title('Transmitted Velocity')
plt.title('Actual Velocity Samples')
# plt.title('ZOH Predictive Schemes')
# plt.title('FOLP Predictive Schemes')
ax.set_xlabel('Time ms')
ax.set_ylabel('Velocity m/s')
ax.scatter(x, A, c='b', s=20, alpha=0.5)
# for i in range(0, len(D0)):
#     plt.plot(D0[i], F0[i], color='r')
# for i in range(0, len(D1)):
#     plt.plot(D1[i], F1[i], color='r')
for i in range(0, len(D)):
    plt.plot(D[i], F[i], color='r')

for j in range(0, len(G)):
    plt.plot(H[j], G[j], color='r')
# for j in range(0, len(G)):
#     plt.plot(H[j], G[j], color='r', linestyle='dashed')
# for j in range(0, len(G0)):
#     plt.plot(H0[j], G0[j], color='r', linestyle='dashed')
# for j in range(0, len(G1)):
#     plt.plot(H1[j], G1[j], color='r', linestyle='dashed')
plt.xticks(np.arange(0, 11, 1.0))
plt.axis([-1, 11, 0, 3.2])
plt.grid()
plt.show()
