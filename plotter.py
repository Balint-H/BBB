import matplotlib.pyplot as plt
import numpy as np
import csv
import seaborn as sns
import pickle


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

with open('MCrets', "rb") as f:
    r = pickle.load(f)
N = 1000
mean_r = np.mean(r, axis=0)
ma = np.convolve(mean_r, np.ones((N,))/N, mode='valid')
d = len(mean_r)-len(ma)
x2 = np.arange(d/2, len(mean_r)-d/2)
rw = rolling_window(mean_r, 1000)
stds = np.std(rw, -1)
sns.set()
#plt.plot(np.convolve(r_sum, np.ones((N,))/N, mode='valid'))

plt.plot(x2, ma,color=sns.color_palette()[0], linewidth=0.7)
plt.fill_between(x2, ma-stds, ma+stds, color=sns.color_palette()[7], alpha=0.2)
plt.title("Moving Average and Moving Standard Deviation of Mean Return (kernel size 1000)")
plt.ylabel("Mean reward returned")
plt.xlabel("Episode")

plt.show()