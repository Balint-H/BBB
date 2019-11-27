import numpy as np
import matplotlib.pyplot as plt
import pickle
from monte_hall_2 import MonteCarlo
from EZ21 import EZ21
import seaborn as sns
import pandas as pd
from decimal import Decimal
from sarsa import SARSA
from q_learn import QLearn
from matplotlib.lines import Line2D

with open(r'C:\Source_files\Python\Pantry\MCQs', "rb") as f:
    MQs = pickle.load(f)
with open(r'C:\Source_files\Python\Pantry\SARSA_Qs', "rb") as f:
    SARSAQs = pickle.load(f)
with open(r'C:\Source_files\Python\BBB\QLong_Qs', "rb") as f:
    QLearnQs = pickle.load(f)

Qs = zip(MQs, SARSAQs, QLearnQs)
MC = MonteCarlo(env_in=EZ21())
SRS = SARSA(env_in=EZ21())
QL = QLearn(env_in=EZ21())
MCrews = list()
SARSArews = list()
QRews = list()
for j, (M, S, L) in enumerate(Qs):
    if j >= 50:
        break
    print(j)
    MCrews.append([])
    SARSArews.append([])
    QRews.append([])
    MC.reset(Q_in=M)
    SRS.reset(Q_in=S)
    QL.reset(Q_in=L)
    for i in range(50000):
        state = MC.env.reset()
        while True:
            state, rew, term = MC.env.step(MC.greedy_pol(tuple(state)))
            if term:
                MCrews[j].append(rew)
                break
        state = SRS.env.reset()
        while True:
            state, rew, term = SRS.env.step(MC.greedy_pol(tuple(state)))
            if term:
                SARSArews[j].append(rew)
                break
        state = QL.env.reset()
        while True:
            state, rew, term = QL.env.step(MC.greedy_pol(tuple(state)))
            if term:
                QRews[j].append(rew)
                break
data = pd.DataFrame(np.transpose([np.mean(MCrews, axis=1), np.mean(SARSArews, axis=1), np.mean(QRews, axis=1)]),
                    columns=["MC", "SARSA", "Q Learning"])
with open("Box_data", 'wb') as myfile:
    pickle.dump(data, myfile)
naive_rews = list()
MC.reset()
for i in range(20000):
    state = MC.env.reset()
    while True:
        state, rew, term = MC.env.step(MC.greedy_pol(tuple(state)))
        if term:
            naive_rews.append(rew)
            break

legend_elements = [Line2D([0], [0], color=sns.color_palette()[1], ls='--', label="Naive Policy"),
                   Line2D([0], [0], marker='o', ls="None", alpha=0.8, label="Trained Policies",
                          markerfacecolor=(0.25, 0.25, 0.25), markeredgecolor="None", markersize=8)]

fig = plt.figure()
sns.set()
ax = sns.boxplot(data=data, showfliers=False, width=0.1, )
ax = sns.swarmplot(alpha=0.8, data=data, color=".25", label="Trained Policies")
ax.axhline(np.mean(naive_rews), ls='--', c=sns.color_palette()[1], label="Naive Policy")
ax.legend(handles=legend_elements, loc='lower left')
ax.text(-0.5, np.mean(MCrews), r'$\bar{r}$:' + '{0:.2E}'.format(Decimal(np.mean(MCrews))), fontsize=11)
ax.text(-0.5, np.mean(MCrews) - 0.01, "σ: {0:.2E}".format(Decimal(np.std(np.mean(MCrews, axis=1)))), fontsize=11)
ax.text(0.5, np.mean(SARSArews), r'$\bar{r}$:' + '{0:.2E}'.format(Decimal(np.mean(SARSArews))), fontsize=11)
ax.text(0.5, np.mean(SARSArews) - 0.01, "σ: {0:.2E}".format(Decimal(np.std(np.mean(SARSArews, axis=1)))), fontsize=11)
ax.text(1.5, np.mean(QRews), r'$\bar{r}$:' + '{0:.2E}'.format(Decimal(np.mean(QRews))), fontsize=11)
ax.text(1.5, np.mean(QRews) - 0.01, "σ: {0:.2E}".format(Decimal(np.std(np.mean(QRews, axis=1)))), fontsize=11)
ax.text(1, np.mean(naive_rews)+0.005, r"Naive $\bar{r}$: " + "{0:.2E}".format(Decimal(np.mean(naive_rews))), fontsize=11)
plt.title(r"Mean returns from 5x$10^4$ games of 50 trained policies each", fontsize=14)
plt.ylabel("Mean of rewards")
plt.show()
