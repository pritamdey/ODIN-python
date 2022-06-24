import numpy as np
import matplotlib.pyplot as plt
from ODIN import logistic_fit as fit
from ODIN import simulation_generation_functions as sims

# n_test
ns = np.array([200, 300, 500, 700, 1000])
n_fitting_times = []
n_influence_times = []

for n in ns:
    n_lobe = 5
    nodes_per_lobe = 7
    X, A = sims.gen_connectomes(n, n_lobe, nodes_per_lobe)

    prop_outliers = 0.1
    prop_flip = 0.1
    outliers, A = sims.outlier_by_flip(A, prop_outliers, prop_flip)

    lam = 0.001
    tol = 0

    fit0 = fit.mm_logistic(A, X, lam, tol, print_out=False, max_iterate=200)
    n_fitting_times.append(fit0[3]/200)
    influence0 = fit.influence_measure(fit0, A, X, lam, print_out=False)
    n_influence_times.append(influence0[3])

n_fitting_times = np.array(n_fitting_times)
n_influence_times = np.array(n_influence_times)

# v_test

vs = np.array([30, 40, 50, 70, 100])
l_fitting_times = []
l_influence_times = []

for v in vs:
    n = 500
    n_lobe = 5
    nodes_per_lobe = int(v / (2 * n_lobe))
    X, A = sims.gen_connectomes(n, n_lobe, nodes_per_lobe)

    prop_outliers = 0.1
    prop_flip = 0.1
    outliers, A = sims.outlier_by_flip(A, prop_outliers, prop_flip)

    lam = 0.001
    tol = 0

    fit0 = fit.mm_logistic(A, X, lam, tol, print_out=False, max_iterate=200)
    l_fitting_times.append(fit0[3]/200)
    influence0 = fit.influence_measure(fit0, A, X, lam, print_out=False)
    l_influence_times.append(influence0[3])

ls = vs*(vs-1)/2
l_fitting_times = np.array(l_fitting_times)
l_influence_times = np.array(l_influence_times)

# Values from a previous run. Use to re-generate figures as in paper
#
# ns =  np.array([200, 300, 500, 700, 1000])
# n_fitting_times = np.array([0.04288278341293335, 0.05904877305030823, 0.08939543008804321, 0.12044040441513061, 0.17003809213638305])
# n_influence_times = np.array([7.638826608657837, 11.309874296188354, 18.632965803146362, 25.587684631347656, 36.711358308792114])
#
# vs = np.array([ 30,  40,  50,  70, 100])
# ls = vs*(vs-1)/2
# l_fitting_times = np.array([0.017105615139007567, 0.027415502071380615, 0.04601656436920166, 0.09064661979675293, 0.21774757385253907])
# l_influence_times = np.array([0.4998815059661865, 1.4840350151062012, 4.211251974105835, 18.455199241638184, 73.24408674240112])

plt.rcParams["figure.figsize"] = (5,5)
plt.rcParams.update({'font.size': 16})
plt.scatter(ns, n_fitting_times, color='black')
plt.xlabel("N")
plt.ylabel("Time (seconds)")
bet1, bet0 = np.polyfit(ns, n_fitting_times, 1)
plt.plot(ns, bet1 * ns+ bet0, color='black')
plt.show()
plt.savefig('figures/hd_version/iter_N.tif', bbox_inches='tight', pad_inches=0.05, dpi=350)
plt.clf()

plt.rcParams["figure.figsize"] = (5,5)
plt.rcParams.update({'font.size': 16})
plt.scatter(ns, n_influence_times, color='black')
plt.xlabel("N")
plt.ylabel("Time (seconds)")
bet1, bet0 = np.polyfit(ns, n_influence_times, 1)
plt.plot(ns, bet1 * ns + bet0, color='black')
plt.show()
plt.savefig('figures/hd_version/inf_N.tif', bbox_inches='tight', pad_inches=0.05, dpi=350)
plt.clf()

plt.rcParams["figure.figsize"] = (5,5)
plt.rcParams.update({'font.size': 16})
plt.scatter(ls, l_fitting_times, color='black')
plt.xlabel("L")
plt.ylabel("Time (seconds)")
bet1, bet0 = np.polyfit(ls, l_fitting_times, 1)
plt.plot(ls, bet1 * ls + bet0, color='black')
plt.show()
plt.savefig('figures/hd_version/iter_L.tif', bbox_inches='tight', pad_inches=0.05, dpi=350)
plt.clf()

plt.rcParams["figure.figsize"] = (5,5)
plt.rcParams.update({'font.size': 16})
x = np.arange(0,5001)
plt.scatter(ls, l_influence_times, color='black')
plt.xlabel("L")
plt.ylabel("Time (seconds)")
bet2, bet1, bet0 = np.polyfit(ls, l_influence_times, 2)
plt.plot(x, bet2 * x**2 + bet1 * x + bet0, color='black')
plt.show()
plt.savefig('figures/hd_version/inf_L.tif', bbox_inches='tight', pad_inches=0.05, dpi=350)
plt.clf()