import numpy as np


def jk_error(vec):
    n = len(vec)
    mean = np.mean(vec)
    take_one_mean_list = []
    for i in range(len(vec)):
        take_one_vec = np.copy(vec)
        take_one_vec[i] = 0
        take_one_mean = np.mean(take_one_vec)
        take_one_mean_list.append(take_one_mean)
    take_one_mean = np.array(take_one_mean_list)

    return (np.sqrt(n - 1) / n) * np.sqrt(np.sum((take_one_mean - mean)**2))
