import numpy as np
import ODIN

n = 500
n_lobe = 5
nodes_per_lobe = 7
lobe_locations, hemi_locations, a = ODIN.simulate.sim(n, n_lobe, nodes_per_lobe)

prop_flipped_edges_all = [0.01, 0.02, 0.07, 0.10, 0.15]
specificity = []
sensitivity = []

reps = 10
for i in range(reps):
    for prop in prop_flipped_edges_all:
        outliers, a1 = ODIN.simulate.outlier_by_flip(a, 0.10, prop)
        outDetect = ODIN.ODIN(lam=0.001, maxiter=2000, tol=1e-6)
        outDetect.fit_and_detect_outliers(a=a1, lobes=lobe_locations, hemis=hemi_locations)
        out = outDetect.outliers
        out_or_outlier = np.union1d(out, outliers).shape[0]
        out_and_outlier = np.intersect1d(out, outliers).shape[0]
        specificity.append((n - out_or_outlier) / (n - outliers.shape[0]))
        sensitivity.append(out_and_outlier / outliers.shape[0])

sensitivity = np.array(sensitivity).reshape(reps, 5)
specificity = np.array(specificity).reshape(reps, 5)

avg_sensitivity = np.mean(sensitivity, axis=0)
avg_specificity = np.mean(specificity, axis=0)
