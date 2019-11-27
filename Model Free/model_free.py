import numpy as np
from abc import ABC, abstractmethod


def allmax(a):
    if len(a) == 0:
        return []
    all_ = list([0])
    max_ = a[0]
    for i in range(1, len(a)):
        if a[i] > max_:
            all_ = list([i])
            max_ = a[i]
        elif a[i] == max_:
            all_.append(i)
    return all_


class ModelFree(ABC):

    @abstractmethod
    def __init__(self, env_in, Q_in=None, n_in=20000, explore_cnst=100):
        """
        dict Q[(state, action)] = [Q-value, number of updates]
        """
        self.env = env_in
        self.n = n_in
        self.Q = Q_in if Q_in is not None and isinstance(Q_in, dict) else dict()
        self.explore = explore_cnst
        self.constant_par = [None, None]  # alpha, epsilon
        return

    def reset(self, Q_in=None):
        self.Q = Q_in if Q_in is not None and isinstance(Q_in, dict) else dict()

    @abstractmethod
    def iter_opt(self):
        """
        Iteration based optimisation.
        :return: list of returns per episode
        """
        return

    def get_Q(self, state_in, act_in):
        return self.Q[(state_in, act_in)][0] if (state_in, act_in) in self.Q else 0

    def get_state_count(self, state_in):
        return sum([self.get_state_act_count(state_in, act) for act in self.env.actions])

    def get_state_act_count(self, state_in, act_in):  # returns number of existing updates in current Q-value
        return self.Q[(state_in, act_in)][1] if (state_in, act_in) in self.Q else 1

    def get_V(self, state_in):
        return max([self.get_Q(state_in, act) for act in self.env.actions])

    def greedy_pol(self, state_in):
        greedy_choice = allmax([self.get_Q(state_in, act) for act in self.env.actions])
        return np.random.choice(greedy_choice) if len(greedy_choice) > 1 else greedy_choice[0]

    def get_alpha(self, state_in, act_in):
        return 1/self.get_state_act_count(state_in, act_in) if self.constant_par[0] is None else self.constant_par[0]

    def e_pol(self, state_in):
        epsilon = self.explore / (self.explore + self.get_state_count(state_in) - 2) \
            if self.constant_par[1] is None else self.constant_par[1]
        prob = epsilon / len(self.env.actions)
        probs = [prob for _ in self.env.actions]
        probs[self.greedy_pol(state_in)] += 1 - epsilon
        return np.random.choice(self.env.actions, p=probs)

