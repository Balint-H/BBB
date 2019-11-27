import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle

with open(r'C:\Source_files\Python\BBB\MCStepscurve_rets', "rb") as f:
    r = pickle.load(f)
m = np.mean(r, axis=1)
fig = plt.figure(figsize=(8, 4))
x = np.cumsum([0]+[2500]*4 + [10000]*3 + [50000]*4 + [100000]*2 + [300000])
x=x[:-1]
plt.plot(x, m)
plt.show()
