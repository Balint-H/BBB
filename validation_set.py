from EZ21 import EZ21
from monte_hall_2 import MonteCarlo
import pickle
from sarsa import SARSA
from q_learn import QLearn
import multiprocessing as multi
import itertools
import numpy as np
from copy import deepcopy
from model_free import allmax


def get_Q(Q_in, state_in, act_in):
    return Q_in[(state_in, act_in)][0] if (state_in, act_in) in Q_in else 0


def greedy_pol(Q_in, state_in):
    greedy_choice = allmax([get_Q(Q_in, state_in, act) for act in [0, 1]])
    return np.random.choice(greedy_choice) if len(greedy_choice) > 1 else greedy_choice[0]


def play_rounds(Q_in):
    env = EZ21()
    results = list()
    for i in range(250000):
        state = tuple(env.reset())
        while True:
            act = greedy_pol(Q_in, state)
            _state, rew, term = env.step(act)  # Holding next state in buffer variable
            if term:
                results.append(rew)
                break
            state = tuple(_state)
    return results


def main():
    nProcess = multi.cpu_count()
    name = ["MCSteps", "SSteps", "Qsteps"]# Change this!
    algQs=list()
    algrews=list()
    steps = [250000] * 4
    for i, agent in enumerate([MonteCarlo(env_in=EZ21()), SARSA(env_in=EZ21()), QLearn(env_in=EZ21())]):
        print(name[i])
        Qs = list()
        algrews.append([])
        run_sum = 0
        for eps in steps:
            run_sum += eps
            print(run_sum)
            agent.n = eps
            agent.iter_opt()
            Qs.append(deepcopy(agent.Q))
        with multi.Pool(nProcess) as pool:
            algrews.append(pool.map(play_rounds, algQs[i]))

        with open(name[i] + "_algQs", 'wb') as myfile:
            pickle.dump(algQs, myfile)
        with open(name[i] + "_algrews", 'wb') as myfile:
            pickle.dump(algrews, myfile)
    return


if __name__ == '__main__':
    main()
