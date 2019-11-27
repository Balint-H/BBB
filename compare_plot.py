import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle

with open(r'C:\Source_files\Python\Pantry\QLearn_rets', "rb") as f:
    r1 = pickle.load(f)
with open(r'C:\Source_files\Python\Pantry\QmultiEpsilon_0.1_rets', "rb") as f:
    r2 = pickle.load(f)
with open(r'C:\Source_files\Python\Pantry\Q_multi_epsilon_0.6_rets', "rb") as f:
    r3 = pickle.load(f)
with open(r'C:\Source_files\Python\BBB\Q_multi_epsilon_0_rets', "rb") as f:
    r4 = pickle.load(f)
fig = plt.figure(figsize=(12, 4))

N = 1000
mean_r1 = np.mean(r1, axis=0)
ma1 = np.convolve(mean_r1, np.ones((N,))/N, mode='valid')
mean_r2 = np.mean(r2, axis=0)
ma2 = np.convolve(mean_r2, np.ones((N,))/N, mode='valid')
mean_r3 = np.mean(r3, axis=0)
ma3 = np.convolve(mean_r3, np.ones((N,))/N, mode='valid')
mean_r4 = np.mean(r4, axis=0)
ma4 = np.convolve(mean_r4, np.ones((N,))/N, mode='valid')
d1 = np.convolve(np.std(r1, axis=0, ddof=1), np.ones((N,))/N, mode='valid')
d2 = np.convolve(np.std(r2, axis=0, ddof=1), np.ones((N,))/N, mode='valid')
d3 = np.convolve(np.std(r3, axis=0, ddof=1), np.ones((N,))/N, mode='valid')
d4 = np.convolve(np.std(r4, axis=0, ddof=1), np.ones((N,))/N, mode='valid')
dif = len(mean_r1)-len(ma1)
x2 = np.arange(dif/2, len(mean_r1)-dif/2)

sns.set()
ax = fig.add_subplot(121)
ax.plot(x2, ma1, linewidth=0.7, alpha=0.9, label=r"Dynamic $\epsilon$")
ax.plot(x2, ma2, linewidth=0.7, alpha=0.9, label=r"$\epsilon=0.1$")
ax.plot(x2, ma3, linewidth=0.7, alpha=0.9, label=r"$\epsilon=0.6$")
ax.plot(x2, ma4, linewidth=0.7, alpha=0.9, label=r"$\epsilon=0$")
plt.title("MA smoothed learning curve for Q-Learning, 90 runs each")
plt.xlabel("Episode")
plt.ylabel("Mean Return")
plt.legend(loc="lower right")
ax = fig.add_subplot(122)
ax.plot(x2, d1, linewidth=0.7, alpha=0.9, label=r"Dynamic $\epsilon$")
ax.plot(x2, d2, linewidth=0.7, alpha=0.9, label=r"$\epsilon=0.1$")
ax.plot(x2, d3, linewidth=0.7, alpha=0.9, label=r"$\epsilon=0.6$")
ax.plot(x2, d4, linewidth=0.7, alpha=0.9, label=r"$\epsilon=0$")
plt.title("MA smoothed $\sigma$ for Q-Learning, 90 runs each")
plt.xlabel("Episode")
plt.ylabel("Standard Deviation")
plt.legend(loc="lower right")
plt.show()
