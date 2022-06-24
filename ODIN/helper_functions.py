import numpy as np


def level_matrix(num):
    level_mat = np.zeros((num, num), dtype='int32')
    s = 1
    for i in range(num):
        for j in range(i + 1):
            level_mat[i, j] = s
            level_mat[j, i] = s
            s = s + 1
    return level_mat


# FUNCTION TO CREATE DESIGN MATRIX FROM METADATA ABOUT NODES
#
# make_lobe_hemi_matrix IN THE SIMULATION GENERATOR GENERATES
# ONE SUCH MATRIX
def make_x(lobe_hemi_matrix):
    v = np.shape(lobe_hemi_matrix)[0]
    hem_names = list(set(lobe_hemi_matrix["Hemisphere"]))
    hem_names.sort()
    lobe_names = list(set(lobe_hemi_matrix["Lobe"]))
    lobe_names.sort()
    n_hemi = len(hem_names)
    n_lobe = len(lobe_names)
    l = int(v * (v - 1) / 2)
    hemi = level_matrix(n_hemi)
    lobe = level_matrix(n_lobe)
    x1 = np.zeros((l, int(hemi.max())), dtype='int32')
    x2 = np.zeros((l, int(lobe.max())), dtype='int32')
    i = 0
    for co in range(1, v):
        for ro in range(co):
            i = i + 1
            hemi1 = hem_names.index(list(lobe_hemi_matrix["Hemisphere"])[co])
            hemi2 = hem_names.index(list(lobe_hemi_matrix["Hemisphere"])[ro])
            x1[i - 1, hemi[hemi1, hemi2] - 1] = 1
            lobe1 = lobe_names.index(list(lobe_hemi_matrix["Lobe"])[co])
            lobe2 = lobe_names.index(list(lobe_hemi_matrix["Lobe"])[ro])
            x2[i - 1, lobe[lobe1, lobe2] - 1] = 1
    x = np.concatenate((x1, x2), axis=1)[:, 1:]
    return x


# THESE FUNCTIONS HELP MOVE BETWEEN MATRIX AND VECTOR
# REPRESENTATIONS OF THE ADJACENCY MATRIX (CONSIDERING ZERO DIAGONALS)
#
# CONVERT VECTOR TO MATRIX FOR DISPLAY
def vector2matrix(x):
    x = np.array(x)
    p = x.shape[0]
    n = int(np.ceil(np.sqrt(2*p)))
    t = 0
    s = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            s[i, j] = x[t]
            s[j, i] = x[t]
            t += 1
    return s


# CONVERT MATRIX TO VECTOR FOR FITTING
def matrix2vector(A):
    A = np.array(A)
    v = A.shape[0]
    l = int(v*(v-1)/2)
    temp = np.zeros(l)
    k = 0
    for i in range(1, v):
        for j in range(i):
            temp[k] = A[i, j]
            k = k + 1
    return temp

