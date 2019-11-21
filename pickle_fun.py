import numpy as np
import matplotlib.pyplot as plt
import pickle
from monte_hall_2 import MonteCarlo
from EZ21 import EZ21
import seaborn as sns
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

with open('QPickle', "rb") as f:
    Qs = pickle.load(f)

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
            V_MC[row, col] = MC.get_V((row, col))
            Pol[row, col] = MC.greedy_pol((row, col))
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


ax = fig.add_subplot(1,2,1, projection='3d')

plt.xlabel("Dealer shows")
ax.set_zlabel("Value")
surf = ax.plot_surface(ss1, ss2, V_MC, cmap=cm.coolwarm, alpha=0.9, zorder=-10)
ax.scatter(hit_x, hit_y, hit_z, marker='o', zorder=1, color='darkorange')
plt.ylabel("Sum in hand")
plt.title("Value estimation from a single MC")

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
            stick_z.append(np.mean(V_MCs, axis=0)[row, col])
        else:
            hit_x.append(row+1)
            hit_y.append(col+1)
            hit_z.append(np.mean(V_MCs, axis=0)[row, col])


ax = fig.add_subplot(1,2,2, projection='3d')
plt.title("Average value from 30 MC")
plt.xlabel("Dealer shows")
surf = ax.plot_surface(ss1, ss2, np.mean(V_MCs, axis=0), cmap=cm.coolwarm, alpha=0.9, zorder=-1)
ax.scatter(hit_x, hit_y, hit_z, marker='o', color='darkorange', zorder=1)

plt.ylabel("Sum in hand")
ax.set_zlabel("Value")



plt.show()

'''
rews=list()
for j, Q in enumerate(Qs):
    MC.reset(Q_in=Q)
    rews.append([])
    #print(j)
    for i in range(10000):
        state = MC.env.reset()
        while True:
            state, rew, term = EZ.step(MC.greedy_pol(tuple(state)))
            if term:
                rews[j].append(rew)
                break

naive_rews = list()
MC.reset()
for i in range(10000):
    state = MC.env.reset()
    while True:
        state, rew, term = EZ.step(MC.greedy_pol(tuple(state)))
        if term:
            naive_rews.append(rew)
            break

data = {"Mean return": np.mean(rews, axis=1)}
sns.set()
data = pd.DataFrame(data)
ax = sns.boxplot(y="Mean return", data=data, showfliers = False, width=0.1)
ax = sns.swarmplot(alpha=0.8, y="Mean return", data=data, color=".25", label="Trained Policies")
ax.axhline(np.mean(naive_rews), ls='--', c=sns.color_palette()[1], label="Naive Policy")
ax.legend(loc='upper right')
ax.text(-0.3, np.mean(rews), "Mean: {0:.3f}".format(np.mean(rews)), fontsize=11)
ax.text(-0.3, np.mean(rews)-0.01, "σ: {0:.3f}".format(np.std(np.mean(rews, axis=1))), fontsize=11)
ax.text(-0.3, np.mean(naive_rews)+0.005, "Naive Mean: {0:.3f}".format(np.mean(naive_rews)), fontsize=11)
plt.title(r"Mean returns from $10^4$ games of 30 trained policies", fontsize=14)
tips = sns.load_dataset("tips")
plt.show()
print('end')
'''
