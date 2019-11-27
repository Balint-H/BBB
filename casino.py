
from EZ21 import EZ21
from monte_hall_2 import MonteCarlo
import pickle
from sarsa import SARSA
from q_learn import QLearn
import time
import multiprocessing as multi
from functools import partial
import itertools



def Off_Pol(cnstnts=(None, None)):
    if "1" in multi.current_process().name:
        print(('%s began working' % multi.current_process().name))
    agent = QLearn(env_in=EZ21(), n_in=80000, cnst_par=cnstnts)
    return agent.iter_opt(), agent.Q


def On_Pol(cnstnts=(None, None)):
    if "1" in multi.current_process().name:
        print(('%s began working' % multi.current_process().name))
    agent = SARSA(env_in=EZ21(), n_in=80000, cnst_par=cnstnts)
    return agent.iter_opt(), agent.Q


def Monte(cnstnts=(None, None)):
    if "1" in multi.current_process().name:
        print(('%s began working' % multi.current_process().name))
    agent = MonteCarlo(env_in=EZ21(), n_in=100000)
    return agent.iter_opt(), agent.Q

def continue_MC(Q_in):
    if "1" in multi.current_process().name:
        print(('%s began working' % multi.current_process().name))
    agent = MonteCarlo(Q_in=Q_in, env_in=EZ21(), n_in=120000)
    return agent.iter_opt(), agent.Q

def continue_Q(Q_in):
    if "1" in multi.current_process().name:
        print(('%s began working' % multi.current_process().name))
    agent = QLearn(Q_in=Q_in, env_in=EZ21(), n_in=100000)
    return agent.iter_opt(), agent.Q

def main():
    with open(r'C:\Source_files\Python\BBB\QLong_Qs', "rb") as f:
        Qs = pickle.load(f)
    nProcess = multi.cpu_count()
    Nrepeats = 90
    name = "Qlongadd"  # Change this!
    f = continue_Q # Also change this!
    with multi.Pool(nProcess) as pool:
        results = pool.map(f, Qs)
    # with multi.Pool(nProcess) as pool:
    #     c = itertools.repeat((None, None), Nrepeats)  # And change this
    #     results = pool.map(f, c)
    unzipped_results = list(zip(*results))
    with open(name+"_rets", 'wb') as myfile:
        pickle.dump(unzipped_results[0], myfile)
    with open(name+"_Qs", 'wb') as myfile:
        pickle.dump(unzipped_results[1], myfile)

    return


if __name__ == '__main__':
    main()
