import numpy as np


def matricize(x):
    x = np.array(x)
    l = x.shape[0]
    r = int(np.ceil(np.sqrt(2*l)))
    t = 0
    s = np.zeros((r, r))
    for i in range(r):
        for j in range(i):
            s[i, j] = x[t]
            s[j, i] = x[t]
            t += 1
    return s


def half_vectorize(a):
    a = np.array(a)
    v = a.shape[0]
    l = int(v*(v-1)/2)
    temp = np.zeros(l)
    k = 0
    for i in range(v):
        for j in range(i):
            temp[k] = a[i, j]
            k = k + 1
    return temp


def matricize_all(a):
    l, n = a.shape
    r = int(np.ceil(np.sqrt(2*l)))
    a_temp = np.zeros((r, r, n))
    for i in range(n):
        a_temp[:, :, i] = matricize(a[:, i])
    return a_temp


def half_vectorize_all(a):
    r = a.shape[0]
    n = a.shape[2]
    l = int(r*(r-1)/2)
    a_temp = np.zeros((l, n))
    for i in range(n):
        a_temp[:, i] = half_vectorize(a[:, :, i])
    return a_temp
