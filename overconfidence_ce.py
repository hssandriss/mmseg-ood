import numpy as np
import matplotlib.pyplot as plt

# ra_ce_aupr = [31.01, 28.08, 27.94, 21.17, 21.1, 21.64, 21.85, 25.1, 22.63, 22.29, 23.19, 20.94, 24.08, 23.20]
# ra_ce_fpr = [53.14, 54.50, 58.62, 60.29, 60.79, 54.80, 59.15, 55.85, 60.55, 61.27, 61.01, 65.25, 58., 61.73]
epochs = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])

ra_ce_aupr = np.array([32.41, 21.57, 28.68, 22.63, 26.43, 21.29, 19.6, 24., 19.28, 20.69, 27.36, 23.31, 25.68, 24.2])
ra_ce_fpr = np.array([45.42, 59.03, 45.01, 55.01, 52.45, 60.11, 65.32, 55.87, 64.58, 64.37, 51.22, 60.33, 56.62, 60.22])

ra_ema_aupr = np.array([32.41, 21.57, 31.32, 29.26, 23.13, 22.35, 23.73, 23.38, 22.17, 23.34, 23.73, 23.52, 24.21, 24.02])
ra_ema_fpr = np.array([45.42, 59.03, 46.03, 44.97, 65.85, 59.79, 61.06, 59.4, 59.48, 62.14, 55.79, 62.93, 61.15, 58.13])

ra_self_dist_aupr = np.array([24.13, 23.7, 21.16, 19.42, 29.83, 22.26, 22.97, 23.87, 20.14, 16.28, 20.48, 19.12, 17.87, 19.22])
ra_self_dist_fpr = np.array([59.83, 71.34, 76.62, 85.24, 77.2, 80.7, 79.73, 83.56, 85.75, 87.5, 82.85, 84.71, 86.62, 85.82])


fig = plt.figure()

plt.plot(epochs, ra_ce_aupr, '-o', label="CE", linewidth=1.)
plt.plot(epochs, ra_ema_aupr, '-o', label="CE+EMA", linewidth=1.)
# plt.plot(epochs, ra_self_dist_aupr, '-o', label="CE+Self-Dist")

plt.legend()

plt.xlabel('Epochs')
plt.ylabel('AUPR (%)')

plt.ylim([10., 40.])
plt.xlim([0, 75])

plt.savefig(f'aupr.png', dpi=300,bbox_inches='tight')


plt.cla()
plt.clf()

fig = plt.figure()

plt.plot(epochs, ra_ce_fpr, '-x', label="CE", linewidth=1.)
plt.plot(epochs, ra_ema_fpr, '-x', label="CE+EMA", linewidth=1.)
# plt.plot(epochs, ra_self_dist_fpr, '-x', label="CE+Self-Dist")

plt.legend()

plt.xlabel('Epochs')
plt.ylabel('FPR95 (%)')

plt.ylim([30., 100.])
plt.xlim([0, 75])

plt.savefig(f'fpr.png', dpi=300,bbox_inches='tight')
