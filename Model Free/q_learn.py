import numpy as np
from model_free import ModelFree


class QLearn(ModelFree):

    def __init__(self, env_in, Q_in=None, n_in=20000, explore_cnst=100, gamma_in=0.9, cnst_par=(None, None)):
        """
        dict Q[(state, action)] = [Q-value, number of updates]
        """
        self.env = env_in
        self.n = n_in
        self.Q = Q_in if Q_in is not None and isinstance(Q_in, dict) else dict()
        self.explore = explore_cnst
        self.gamma = gamma_in
        self.constant_par = cnst_par

    def iter_opt(self):
        ret = list()
        for i in range(self.n):
            state = tuple(self.env.reset())
            while True:
                act = self.e_pol(state)
                _state, rew, term = self.env.step(act) # Holding next state in buffer variable
                _state = tuple(_state)
                _act = self.greedy_pol(_state)

                self.Q[state, act] = [self.get_Q(state, act) +
                                      self.get_alpha(state, act) *
                                      (rew - self.get_Q(state, act) + self.gamma*self.get_Q(_state, _act)),
                                      self.get_state_act_count(state, act) + 1]
                if term:
                    ret.append(rew)
                    break
                state = tuple(_state)
        return ret