import numpy as np
import ODIN
import matplotlib.pyplot as plt


def box_plot_sims(pro):
    n = 500
    n_lobe = 5
    nodes_per_lobe = 7
    lobe_locations, hemi_locations, a = ODIN.simulate.sim(n, n_lobe, nodes_per_lobe)
    outliers, a = ODIN.simulate.outlier_by_flip(a, 0.10, pro/100)
    outDetect = ODIN.ODIN(lam=0.001, maxiter=200, tol=0)
    outDetect.fit_and_detect_outliers(a=a, lobes=lobe_locations, hemis=hemi_locations)
    im1_out = outDetect.im1[outliers]
    im1_in = outDetect.im1[np.setdiff1d(np.arange(n), outliers)]
    return [im1_out, im1_in, pro]


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