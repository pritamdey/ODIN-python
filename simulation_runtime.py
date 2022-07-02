import numpy as np
import matplotlib.pyplot as plt
import ODIN

# n_test
ns = np.array([200, 300, 500, 700, 1000])
n_fitting_times = []
n_influence_times = []

for n in ns:
    n_lobe = 5
    nodes_per_lobe = 7

    lobe_locations, hemi_locations, a = ODIN.simulate.sim(n, n_lobe, nodes_per_lobe)

    outDetect = ODIN.ODIN(lam=0.001, maxiter=200, tol=0)
    outDetect.fit_and_detect_outliers(a=a, lobes=lobe_locations, hemis=hemi_locations)
    n_fitting_times.append(outDetect.time[0]/200)
    n_influence_times.append(outDetect.time[1])

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
    lobe_locations, hemi_locations, a = ODIN.simulate.sim(n, n_lobe, nodes_per_lobe)

    outDetect = ODIN.ODIN(lam=0.001, maxiter=200, tol=0)
    outDetect.fit_and_detect_outliers(a=a, lobes=lobe_locations, hemis=hemi_locations)

    l_fitting_times.append(outDetect.time[0]/200)
    l_influence_times.append(outDetect.time[1])

ls = vs*(vs-1)/2
l_fitting_times = np.array(l_fitting_times)
l_influence_times = np.array(l_influence_times)

plt.rcParams["figure.figsize"] = (12, 9)
plt.rcParams.update({'font.size': 16})
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.subplots_adjust(bottom=0.1, top=0.95, hspace=0.35, wspace=0.25)

ax1.scatter(ns, n_fitting_times, color='black')
ax1.set_xlabel("N")
ax1.set_ylabel("Time (seconds)")
bet1, bet0 = np.polyfit(ns, n_fitting_times, 1)
ax1.plot(ns, bet1 * ns+ bet0, color='black')

ax2.scatter(ns, n_influence_times, color='black')
ax2.set_xlabel("N")
ax2.set_ylabel("Time (seconds)")
bet1, bet0 = np.polyfit(ns, n_influence_times, 1)
ax2.plot(ns, bet1 * ns + bet0, color='black')

ax3.scatter(ls, l_fitting_times, color='black')
ax3.set_xlabel("L")
ax3.set_ylabel("Time (seconds)")
bet1, bet0 = np.polyfit(ls, l_fitting_times, 1)
ax3.plot(ls, bet1 * ls + bet0, color='black')

x = np.arange(0,5001)
ax4.scatter(ls, l_influence_times, color='black')
ax4.set_xlabel("L")
ax4.set_ylabel("Time (seconds)")
bet2, bet1, bet0 = np.polyfit(ls, l_influence_times, 2)
ax4.plot(x, bet2 * x**2 + bet1 * x + bet0, color='black')

plt.show()