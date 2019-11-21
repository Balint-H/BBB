import numpy as np
from EZ21 import EZ21
from monte_hall_2 import MonteCarlo
import matplotlib.pyplot as plt
import csv
from copy import deepcopy
import pickle


MC = MonteCarlo(env_in=EZ21(), n_in=80000)
rets = list()
MCQs = list()
for i in range(90):
    print(i)
    rets.append(MC.iter_opt())
    MC.reset()
with open("MCrets", 'wb') as myfile:
    pickle.dump(rets, myfile)



