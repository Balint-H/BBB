import numpy as np
from model_free import ModelFree


class MonteCarlo(ModelFree):

    def __init__(self, env_in, Q_in=None, n_in=20000, explore_cnst=100, cnst_par=(None, None)):
        """
        dict Q[(state, action)] = [Q-value, number of updates]
        """
        self.env = env_in
        self.n = n_in
        self.Q = Q_in if Q_in is not None and isinstance(Q_in, dict) else dict()
        self.explore = explore_cnst
        self.constant_par = cnst_par
        self.Returns = dict()

    def iter_opt(self):
        ret = list()
        for i in range(self.n):
            state = tuple(self.env.reset())
            trace = list()
            while True:
                act = self.e_pol(state)
                _state, rew, term = self.env.step(act)  # Holding next state in buffer variable
                trace.append((state, act, rew))
                if term:
                    ret.append(rew)
                    break
                state = tuple(_state)
            for state, act, rew in trace:
                self.Q[state, act] = [
                    self.get_Q(state, act) + self.get_alpha(state, act)*(rew - self.get_Q(state, act)),
                    self.get_state_act_count(state, act) + 1]
        return ret




