import pickle
import multiprocessing as multi
import numpy as np
from model_free import allmax
from EZ21 import EZ21

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
    with open(r'C:\Source_files\Python\BBB\Qsteps_algQs', "rb") as f:
        q = pickle.load(f)

    nProcess = multi.cpu_count()
    rews=list()
    for i, alg in enumerate(q):
        print(i)
        with multi.Pool(nProcess) as pool:
            rews.append(pool.map(play_rounds, alg))

    with open("extracted_res", 'wb') as myfile:
        pickle.dump(rews, myfile)


if __name__ == '__main__':
    main()