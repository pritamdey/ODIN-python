import numpy as np
from ODIN import logistic_fit as fit
from ODIN import simulation_generation_functions as sims
import matplotlib.pyplot as plt


def box_plot_sims(pro):
    n = 500
    n_lobe = 5
    nodes_per_lobe = 7
    x, a = sims.gen_connectomes(n, n_lobe, nodes_per_lobe)
    outlier, a1 = sims.outlier_by_flip(a, 0.10, pro/100)
    lam = 0.001
    tol = 1e-6
    fit0 = fit.mm_logistic(a1, x, lam, tol, print_out=False, max_iterate=200)
    influence0 = fit.influence_measure(fit0, a1, x, lam, print_out=False)
    influence1_out = influence0[1][outlier]
    influence1_in = influence0[1][np.setdiff1d(np.arange(n), outlier)]
    return [influence1_out, influence1_in, pro]


def plot_box_plot(infl):
    pro = infl[2]
    infl = infl[0:2]
    plt.clf()
    plt.rcParams["figure.figsize"] = (5, 5)
    plt.rcParams.update({'font.size': 16})
    plt.boxplot(infl, notch=False,
                labels=["Outliers", "Non-outliers"], vert=True, widths=0.35,
                patch_artist=True, boxprops=dict(facecolor="grey", color="black"),
                medianprops=dict(color="black", lw=3))
    plt.ylabel("IM$_1$")
    plt.show()
    plt.savefig(f'figures/hd_version/box_{pro}.tif', bbox_inches='tight', pad_inches=0.05, dpi=350)
    return True


prop = 1
influences_1 = box_plot_sims(prop)

prop = 5
influences_5 = box_plot_sims(prop)

prop = 10
influences_10 = box_plot_sims(prop)

plot_box_plot(influences_1)
plot_box_plot(influences_5)
plot_box_plot(influences_10)