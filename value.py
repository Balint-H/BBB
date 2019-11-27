import numpy as np
import matplotlib.pyplot as plt
import pickle
from monte_hall_2 import MonteCarlo
from EZ21 import EZ21
import seaborn as sns
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time

method = "MC"

with open(r'C:\Source_files\Python\BBB\MC100000_Qs', "rb") as f:
    Qs = pickle.load(f)
with open(r'C:\Source_files\Python\Pantry\MCQs', "rb") as f:
    Qs += tuple(pickle.load(f))

s1 = np.arange(10)+1
s2 = np.arange(21)+1
ss1, ss2 = np.meshgrid(s1, s2, indexing='ij')

EZ = EZ21()
MC = MonteCarlo(env_in=EZ)
fig = plt.figure(figsize=(10, 6))
V_MCs = list()
Pols = list()
Pol = []
V_MC=[]
for j, Q in enumerate(Qs):
    MC.reset(Q)
    V_MC = np.zeros([10, 21])
    Pol = np.zeros([10, 21])
    for row in range(10):
        for col in range(21):
            V_MC[row, col] = MC.get_V((row+1, col+1))
            Pol[row, col] = MC.greedy_pol((row+1, col+1))
    V_MCs.append(V_MC)
    Pols.append(Pol)

mPol = np.round(np.mean(Pols, axis=0))

stick_x=list()
stick_y=list()
stick_z=list()
hit_x=list()
hit_y=list()
hit_z=list()
for row in range(10):
    for col in range(21):
        if not Pol[row, col]:
            stick_x.append(row+1)
            stick_y.append(col+1)
            stick_z.append(V_MC[row, col])
        else:
            hit_x.append(row+1)
            hit_y.append(col+1)
            hit_z.append(V_MC[row, col])
row=5
col=-10
hit_x.append(row+1)
hit_y.append(col+1)
hit_z.append(V_MC[row, col]+5)

ax = fig.add_subplot(1,2,1, projection='3d')

ax.set_xlim3d(-0.1, 10.1)
ax.set_ylim3d(-0.1,21.1)
ax.set_zlim3d(-0.38,1)

plt.title("Value estimation from a single "+method)
plt.xlabel("Dealer shows")
ax.set_zlabel("Value")
plt.ylabel("Sum in hand")

surf = ax.plot_surface(ss1, ss2, V_MC, cmap=cm.coolwarm, alpha=0.75, edgecolor=(0.4, 0.4, 0.4))
ax.scatter(hit_x+hit_x, hit_y+hit_y, hit_z+hit_z, marker='o', color='darkorange')
stick_x=list()
stick_y=list()
stick_z=list()
hit_x=list()
hit_y=list()
hit_z=list()
for row in range(10):
    for col in range(21):
        if not mPol[row, col]:
            stick_x.append(row+1)
            stick_y.append(col+1)
            stick_z.append(np.mean(V_MCs, axis=0)[row, col])
        else:
            hit_x.append(row+1)
            hit_y.append(col+1)
            hit_z.append(np.mean(V_MCs, axis=0)[row, col])
row=5
col=-10
hit_x.append(row+1)
hit_y.append(col+1)
hit_z.append(V_MC[row, col]+5)

ax = fig.add_subplot(1,2,2, projection='3d')
ax.set_xlim3d(-0.1, 10.1)
ax.set_ylim3d(-0.1,21.1)
ax.set_zlim3d(-0.38,1)
surf = ax.plot_surface(ss1, ss2, np.mean(V_MCs, axis=0), cmap=cm.coolwarm, alpha=0.75, edgecolor=(0.4, 0.4, 0.4))
ax.scatter(hit_x+hit_x, hit_y+hit_y, hit_z+hit_z, marker='o', color='darkorange')
fig.colorbar(surf, ax=fig.axes)

plt.title("Average value from 90 "+method)
plt.ylabel("Sum in hand")
ax.set_zlabel("Value")
plt.xlabel("Dealer shows")
plt.show()

