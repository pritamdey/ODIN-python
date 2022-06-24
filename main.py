import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2, kde
from ODIN import ODIN as fit
from ODIN import helper_functions as helper
data_path = 'data/'

# This stores the biological structure of the connectome
x_meta = pd.read_csv(data_path+'LobeHemiMatrix.csv')
X = helper.make_x(x_meta)

# The connectomes are stored as vectors of the
# lower triangular part of the adjacency matrix row by row
A = np.array(pd.read_csv(data_path+'A.csv', dtype='int32'))
N = A.shape[1]

#####################################################################################################
#####################################################################################################
# Fitting the model and calculating influence
lam = 0.001
tol = 5e-6
clean_fit = fit.mm_logistic(A, X, lam, tol, print_out=True, max_iterate=10000)
influence_clean = fit.influence_measure(clean_fit, A, X, lam)

#####################################################################################################
#####################################################################################################
# Outliers and Inliers

# First influence measure threshold
med = np.median(influence_clean[1])
quartiles = np.quantile(influence_clean[1], [0.25, 0.75], interpolation='lower')
iqr = quartiles[1] - quartiles[0]
threshold1 = med + 1.5 * iqr
out1 = np.where(influence_clean[1] > threshold1)

# Second influence measure threshold
threshold2 = chi2.ppf(0.999, 23)
out2 = np.where(influence_clean[2] > threshold2)

out = np.union1d(out1, out2)
ins = np.setdiff1d(np.arange(N), out)

#####################################################################################################
#####################################################################################################
# Influence Plots

plt.clf()
plt.rcParams["figure.figsize"] = (20, 10)
plt.rcParams.update({'font.size': 16})
plt.plot(np.arange(N), influence_clean[1], color='black', marker='o', linestyle='', markersize=1)
plt.title("Influence Measure 1")
plt.xlabel("index")
plt.ylabel("IM$_1$(index)")
plt.axhline(y=threshold1, color='red', linestyle='-', lw=3)
plt.show()

plt.clf()
plt.rcParams["figure.figsize"] = (20, 10)
plt.rcParams.update({'font.size': 16})
plt.plot(np.arange(N), influence_clean[2], color='black', marker='o', linestyle='', markersize=1)
plt.title("Influence Measure 2")
plt.xlabel("index")
plt.ylabel("IM$_2$(index)")
plt.axhline(y=threshold2, color='red', linestyle='-', lw=3)
plt.show()


plt.clf()
plt.rcParams["figure.figsize"] = (20, 10)
plt.rcParams.update({'font.size': 16})
plt.plot(influence_clean[1], influence_clean[2], 'bo')
plt.title("Two influence measures")
plt.xlabel("IM$_1$")
plt.ylabel("IM$_2$")
plt.show()

#####################################################################################################
#####################################################################################################
# No. of edges plots

intra_hemisphere_edges = np.where(X[:, 0] == 1)[0]
inter_hemisphere_edges = np.where(X[:, 0] == 0)[0]
a1 = np.sum(A[intra_hemisphere_edges, :], 0).reshape(N)
a2 = np.sum(A[inter_hemisphere_edges, :], 0).reshape(N)

x1 = np.arange(0, 800, 0.1)
density_intra_out = kde.gaussian_kde(a1[out])(x1)
density_intra_in = kde.gaussian_kde(a1[ins])(x1)

x2 = np.arange(250, 1150, 0.1)
density_inter_out = kde.gaussian_kde(a2[out])(x2)
density_inter_in = kde.gaussian_kde(a2[ins])(x2)

plt.clf()
plt.rcParams["figure.figsize"] = (5, 5)
plt.rcParams.update({'font.size': 16})
plt.plot(x1, density_intra_out)
plt.plot(x1, density_intra_in)
plt.xlabel("Number of edges")
plt.ylabel("Density")
# plt.title("Number of edges connecting\nROIs between hemispheres")
plt.show()
plt.savefig(f'figures/hd_version/edges_between.tif', bbox_inches='tight', pad_inches=0.05, dpi=350)

plt.clf()
plt.rcParams["figure.figsize"] = (5, 5)
plt.rcParams.update({'font.size': 16})
plt.plot(x2, density_inter_out)
plt.plot(x2, density_inter_in)
plt.xlabel("Number of edges")
plt.ylabel("Density")
# plt.title("Number of edges connecting\nROIs within hemispheres")
plt.show()
plt.savefig(f'figures/hd_version/edges_within.tif', bbox_inches='tight', pad_inches=0.05, dpi=350)

#####################################################################################################
#####################################################################################################
# Plot of adjacency matrix

# helper.show_connectome(A[:, 0])
