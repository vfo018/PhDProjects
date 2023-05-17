import numpy as np
from scipy.spatial import distance


def pd_codecs(v1, threshold):
    v_t = [v1[0]]
    v0 = v1[0]
    for i, v in enumerate(v1):
        if distance.cdist(np.array([v]), np.array([v0]), 'euclidean') > threshold * np.linalg.norm(v):

