import numpy as np


def level_matrix(num):
    level_mat = np.zeros((num, num), dtype='int32')
    ss = 1
    for ii in range(num):
        for jj in range(ii + 1):
            level_mat[ii, jj] = ss
            level_mat[jj, ii] = ss
            ss += 1
    return level_mat


def make_x(lobes, hemis):
    v = len(lobes)
    hem_names = list(set(hemis))
    hem_names.sort()
    lobe_names = list(set(lobes))
    lobe_names.sort()
    n_hemi = len(hem_names)
    n_lobe = len(lobe_names)
    l = int(v * (v - 1) / 2)
    hemi = level_matrix(n_hemi)
    lobe = level_matrix(n_lobe)
    x1 = np.zeros((l, int(np.max(hemi))), dtype='int32')
    x2 = np.zeros((l, int(np.max(lobe))), dtype='int32')
    i = 0
    for co in range(1, v):
        for ro in range(co):
            i = i + 1
            hemi1 = hem_names.index(list(hemis)[co])
            hemi2 = hem_names.index(list(hemis)[ro])
            x1[i - 1, hemi[hemi1, hemi2] - 1] = 1
            lobe1 = lobe_names.index(list(lobes)[co])
            lobe2 = lobe_names.index(list(lobes)[ro])
            x2[i - 1, lobe[lobe1, lobe2] - 1] = 1
    x = np.concatenate((x1, x2), axis=1)[:, 1:]
    return x
