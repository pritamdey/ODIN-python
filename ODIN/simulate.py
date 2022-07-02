from . import make_x
from .math_functions import *
from .mat_vec_transforms import matricize_all, half_vectorize_all


# GENERATES THE HEMISPHERE AND LOBE LOCATIONS OF THE NODES AND
# n BINARY ADJACENCY MATRICES
#
# RETURNS A LIST CONTAINING THE LOBE DATA, HEMISPHERE DATA AND THE ADJACENCY MATRICES
#
def sim(n, n_lobe=5, nodes_per_lobe=7):
    nodes_per_hemi = n_lobe * nodes_per_lobe
    hemi_locations = ['L'] * nodes_per_hemi + ['R'] * nodes_per_hemi
    lobe0 = [str(int(i / nodes_per_lobe)) for i in range(nodes_per_hemi)]
    lobe_locations = lobe0 + lobe0
    x = make_x.make_x(lobe_locations, hemi_locations)
    l = x.shape[0]
    p = x.shape[1]
    z = np.random.standard_cauchy(size=l)
    bet = np.random.normal(loc=0.0, scale=1.0, size=p*n).reshape(p, n)
    b = expit(z.reshape(l, 1) + np.matmul(x, bet))
    a = matricize_all(np.random.binomial(1, b))
    return [lobe_locations, hemi_locations, a]


# CONTAMINATES RANDOMLY A SAMPLE OF ADJACENCY MATRICES
# BY RANDOMLY FLIPPING A SET PROPORTION OF EDGES IN
# A SET PROPORTION OF INDIVIDUALS
#
# RETURNS A LIST CONTAINING THE INDICES OF THE CONTAMINATED INDIVIDUALS
# AND THE CONTAMINATED SAMPLE OF ADJACENCY MATRICES
#
def outlier_by_flip(a, prop_outliers, prop_flip):
    b = half_vectorize_all(a)
    l = b.shape[0]
    n = b.shape[1]
    n_outliers = int(n * prop_outliers)
    n_flips_per_outlier = int(l * prop_flip)
    outliers = np.sort(np.random.choice(np.arange(n), n_outliers, False))
    for i in outliers:
        edges_to_flip = np.random.choice(np.arange(l), n_flips_per_outlier, False)
        for j in edges_to_flip:
            b[j, i] = 1 - b[j, i]
    return [outliers, matricize_all(b)]
