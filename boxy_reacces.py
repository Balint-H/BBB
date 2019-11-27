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


with open(r'C:\Source_files\Python\BBB\Box_data', "rb") as f:
    data = pickle.load(f)


MC = MonteCarlo(env_in=EZ21())
naive_rews = list()
for i in range(20000):
    state = MC.env.reset()
    while True:
        state, rew, term = MC.env.step(MC.greedy_pol(tuple(state)))
        if term:
            naive_rews.append(rew)
            break


legend_elements = [Line2D([0], [0], marker='o', ls="None", alpha=0.8, label="Trained Policies",
                          markerfacecolor=(0.25, 0.25, 0.25), markeredgecolor="None", markersize=8)]

fig = plt.figure()
sns.set()
ax = sns.boxplot(data=data, showfliers=False, width=0.1, )
ax = sns.swarmplot(alpha=0.8, data=data, color=".25", label="Trained Policies")
#ax.axhline(np.mean(naive_rews), ls='--', c=sns.color_palette()[1], label="Naive Policy")
ax.legend(handles=legend_elements, loc='lower left')
ax.text(-0.5, np.mean(data['MC']), r'$\bar{r}$:' + '{0:.2E}'.format(Decimal(np.mean(data['MC']))), fontsize=11)
ax.text(-0.5, np.mean(data['MC']) - 0.01, "σ: {0:.2E}".format(Decimal(np.std(data['MC']))), fontsize=11)
ax.text(0.5, np.mean(data['SARSA']), r'$\bar{r}$:' + '{0:.2E}'.format(Decimal(np.mean(data['SARSA']))), fontsize=11)
ax.text(0.5, np.mean(data['SARSA']) - 0.01, "σ: {0:.2E}".format(Decimal(np.std(data['SARSA']))), fontsize=11)
ax.text(1.5, np.mean(data['Q Learning']), r'$\bar{r}$:' + '{0:.2E}'.format(Decimal(np.mean(data['Q Learning']))), fontsize=11)
ax.text(1.5, np.mean(data['Q Learning']) - 0.01, "σ: {0:.2E}".format(Decimal(np.std(data['Q Learning']))), fontsize=11)
#ax.text(1, np.mean(naive_rews)+0.005, r"Naive $\bar{r}$: " + "{0:.2E}".format(Decimal(np.mean(naive_rews))), fontsize=11)
plt.title(r"Mean returns from 5x$10^4$ games of 50 trained policies each", fontsize=14)
plt.ylabel("Mean of rewards")
plt.show()