import numpy as np
import pandas as pd
from scipy.stats import cauchy
from . import helper_functions as helper


# GENERATES A LOBE HEMISPHERE METADATA FRAME USED TO CREATE
# THE DESIGN MATRIX IN HELPER
#
def make_lobe_hemi_matrix(n_lobe=5, nodes_per_lobe=7):
    nodes_per_hemi = n_lobe * nodes_per_lobe
    v = 2 * nodes_per_hemi
    hemi = ['L' for i in range(nodes_per_hemi)] + ['R' for i in range(nodes_per_hemi)]
    lobe0 = [str(int(i / nodes_per_lobe)) for i in range(nodes_per_hemi)]
    lobe = lobe0 + lobe0
    data = {'Hemisphere': hemi, 'Lobe': lobe}
    return pd.DataFrame(data)


# GENERATES THE DESIGN MATRIX X AND
# n BINARY ADJACENCY MATRICES, BUT IN THEIR VECTORIZED FORM,
# EACH AS A COLUMN OF A
#
# RETURNS A LIST CONTAINING X AND A
#
def gen_connectomes(n, n_lobe=5, nodes_per_lobe=7):
    def expit(y):
        return np.where(y < 500, np.exp(y) / (1 + np.exp(y)), 1.0)
    x = helper.make_x(make_lobe_hemi_matrix(n_lobe, nodes_per_lobe))
    l = x.shape[0]
    p = x.shape[1]
    z = cauchy.rvs(loc=0, scale=1, size=l)
    bet = np.random.normal(loc=0.0, scale=1.0, size=p*n).reshape(p, n)
    b = expit(z.reshape(l, 1) + np.matmul(x, bet))
    a = np.random.binomial(1, b)
    return [x, a]


# CONTAMINATES RANDOMLY A SAMPLE OF ADJACENCY MATRICES IN THEIR
# VECTOR FORM BY RANDOMLY FLIPPING A SET PROPORTION OF EDGES IN
# A SET PROPORTION OF INDIVIDUALS
#
# RETURNS A LIST CONTAINING THE INDICES OF THE CONTAMINATED INDIVIDUALS
# AND THE CONTAMINATED SAMPLE OF VECTORIZED ADJACENCY MATRICES
#
def outlier_by_flip(a, prop_outliers, prop_flip):
    b = np.copy(a)
    l = b.shape[0]
    n = b.shape[1]
    n_outliers = int(n * prop_outliers)
    n_flips_per_outlier = int(l * prop_flip)
    outliers = np.sort(np.random.choice(np.arange(n), n_outliers, False))
    for i in outliers:
        edges_to_flip = np.random.choice(np.arange(l), n_flips_per_outlier, False)
        for j in edges_to_flip:
            b[j, i] = 1 - b[j, i]
    return [outliers, b]

