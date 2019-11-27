import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


with open(r'Stest_rets', "rb") as f:
    r = pickle.load(f)
fig = plt.figure(figsize=(8, 4))
N = 10000
mean_r = np.mean(r, axis=0)
ma = np.convolve(mean_r, np.ones((N,))/N, mode='valid')
d = len(mean_r)-len(ma)
x2 = np.arange(d/2, len(mean_r)-d/2)
p=np.std(r, axis=0, ddof=1)
pp = np.convolve(p, np.ones((N,))/N, mode='valid')
# pp = np.std(rolling_window(p, 1000), -1)
sns.set()
plt.subplot(121)
#plt.plot(np.convolve(r_sum, np.ones((N,))/N, mode='valid'))
plt.fill_between(x2, ma-pp/3, ma+pp/3, color=sns.color_palette()[7], alpha=0.2, label=r"$\frac{\sigma}{3}$ region (MA)")
plt.plot(x2, ma,color=sns.color_palette()[0], linewidth=0.7, label="Moving Average")
plt.plot(mean_r,color=sns.color_palette()[1], linewidth=0.4, alpha=0.3, label="Mean Return")
plt.legend(loc='lower right')
plt.title(r"Mean Return $\bar{r}$ and MA($\bar{r}$, 1000)")
plt.ylabel("Mean reward returned")
plt.xlabel("Episode")
ax = fig.add_subplot(122)
plt.plot(pp,linewidth= 0.5, label=r"MA($\sigma$, 1000) ")
plt.title(r"Standard Deviation $\sigma$ and MA($\sigma$, 1000)")
plt.ylabel("Standard Deviation")
plt.xlabel("Episode")
plt.legend(loc='lower right')
plt.subplots_adjust(wspace=0.35, bottom=0.15)
plt.show()
