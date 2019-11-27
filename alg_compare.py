import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import pandas as pd

# with open(r'C:\Source_files\Python\BBB\extracted_res', "rb") as f:
#     r = np.array(pickle.load(f))
# m=np.mean
# s=np.std
# print([s(r[t,-1,:])for t in range(3)])
# print([m(r[t,-1,:])for t in range(3)])

with open(r'C:\Source_files\Python\BBB\MC100000_rets', "rb") as f:
    r1 = np.array(pickle.load(f))[:, :80000]
with open(r'C:\Source_files\Python\BBB\MClong_rets', "rb") as f:
    r1 = np.concatenate((r1, pickle.load(f)), axis=1)
with open(r'C:\Source_files\Python\BBB\S200000_rets', "rb") as f:
    r2 = pickle.load(f)
with open(r'C:\Source_files\Python\BBB\QLong_rets', "rb") as f:
    r3 = np.array(pickle.load(f))
with open(r'C:\Source_files\Python\BBB\Qlongadd_rets', "rb") as f:
    r3 = np.concatenate((r3, pickle.load(f)), axis=1)
fig = plt.figure(figsize=(12, 4))

N = 1000
mean_r1 = np.mean(r1, axis=0)
ma1 = np.convolve(mean_r1, np.ones((N,)) / N, mode='valid')
mean_r2 = np.mean(r2, axis=0)
ma2 = np.convolve(mean_r2, np.ones((N,)) / N, mode='valid')
mean_r3 = np.mean(r3, axis=0)
ma3 = np.convolve(mean_r3, np.ones((N,)) / N, mode='valid')
d1 = np.convolve(np.std(r1, axis=0, ddof=1), np.ones((N,)) / N, mode='valid')
d2 = np.convolve(np.std(r2, axis=0, ddof=1), np.ones((N,)) / N, mode='valid')
d3 = np.convolve(np.std(r3, axis=0, ddof=1), np.ones((N,)) / N, mode='valid')
dif = len(mean_r1) - len(ma1)
x2 = np.arange(dif / 2, len(mean_r1) - dif / 2)

sns.set()
ax = fig.add_subplot(121)
ax.plot(x2, ma1, linewidth=0.7, alpha=0.8, label="MC")
ax.plot(x2, ma2, linewidth=0.7, alpha=0.8, label="SARSA")
ax.plot( ma3, linewidth=0.7, alpha=0.8, label="Q Learning")
plt.title("Moving average of the mean return for different algorithms")
plt.xlabel("Episode")
plt.ylabel("Mean Return")
plt.legend(loc="lower right")

plt.legend()

ax = fig.add_subplot(122)

ax.plot(x2, d1, linewidth=0.7, alpha=0.8, label="MC")
ax.plot(x2, d2, linewidth=0.7, alpha=0.8, label="SARSA")
ax.plot(x2, d3, linewidth=0.7, alpha=0.8, label="Q Learning")
plt.title("Moving average of the standard deviation for different algorithms")
plt.xlabel("Episode")
plt.ylabel("Standard Deviation")
plt.legend(loc="lower right")
plt.show()