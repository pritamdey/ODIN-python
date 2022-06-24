import numpy as np
from scipy.stats import chi2
from ODIN import logistic_fit as fit
from ODIN import simulation_generation_functions as sims

n = 500
n_lobe = 5
nodes_per_lobe = 7
x, a = sims.gen_connectomes(n, n_lobe, nodes_per_lobe)

prop_flipped_edges_all = [0.01, 0.02, 0.07, 0.10, 0.15]
specificity = []
sensitivity = []

reps = 10
for i in range(reps):
    for prop in prop_flipped_edges_all:
        outlier, a1 = sims.outlier_by_flip(a, 0.10, prop)
        lam = 0.001
        tol = 1e-6
        fit0 = fit.mm_logistic(a1, x, lam, tol, print_out=False, max_iterate=200)
        influence0 = fit.influence_measure(fit0, a1, x, lam, print_out=False)

        med = np.median(influence0[1])
        quartiles = np.quantile(influence0[1], [0.25, 0.75], interpolation='lower')
        iqr = quartiles[1] - quartiles[0]
        threshold1 = med + 1.5 * iqr
        out1 = np.where(influence0[1] > threshold1)
        threshold2 = chi2.ppf(0.999, 17)
        out2 = np.where(influence0[2] > threshold2)
        out = np.union1d(out1, out2)
        out_or_outlier = np.union1d(out, outlier).shape[0]
        out_and_outlier = np.intersect1d(out, outlier).shape[0]
        specificity.append((n - out_or_outlier) / (n - outlier.shape[0]))
        sensitivity.append(out_and_outlier / outlier.shape[0])

sensitivity = np.array(sensitivity).reshape(reps, 5)
specificity = np.array(specificity).reshape(reps, 5)

avg_sensitivity = np.mean(sensitivity, axis=0)
avg_specificity = np.mean(specificity, axis=0)
