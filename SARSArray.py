import numpy as np

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


def Qid(state_in, act_in):
    return tuple(np.append(state_in, act_in))


class SARSArray:
    def __init__(self, env_in, Q_shape=None, Q_in=None, n_in=20000, explore_cnst=100, gamma_in=0.1):
        """
        dict Q[(state, action)] = [Q-value, number of updates]
        """
        self.env = env_in
        self.n = n_in
        if not isinstance(Q_shape, list):
            print("Supply Q shape as list!")
            raise TypeError
        Q_shape[:-1] = [sh + 1 for sh in Q_shape[:-1]]
        self.Q = Q_in if Q_in is not None and isinstance(Q_in, dict) else np.zeros(Q_shape)
        self.explore = explore_cnst
        self.gamma = gamma_in
        self.visit_count = np.ones(Q_shape)

    def reset(self, Q_in=None, Q_shape=None):
        self.Q = Q_in if Q_in is not None and isinstance(Q_in, dict) else np.zeros(Q_shape)

    def iter_opt(self):
        ret = list()
        for i in range(self.n):
            state = np.array(self.env.reset())
            while True:
                act = self.e_pol(state)
                _state, rew, term = self.env.step(act)  # Holding next state in buffer variable
                _act = self.e_pol(_state)
                self.Q[Qid(state, act)] = self.Q[Qid(state, act)] + \
                                          (rew - self.Q[Qid(state, act)] + self.gamma*self.Q[Qid(_state, _act)])\
                                          / self.visit_count[Qid(state, act)]
                self.visit_count[Qid(state, act)] += 1
                if term:
                    ret.append(rew)
                    break
                state = np.array(_state)
        return ret

    def get_state_count(self, state_in):
        return sum([self.visit_count[Qid(state_in, act)] for act in self.env.actions])

    def get_V(self, state_in):
        return max([self.Q[Qid(state_in, act)] for act in self.env.actions])

    def greedy_pol(self, state_in):
        greedy_choice = allmax([self.Q[Qid(state_in, act)] for act in self.env.actions])
        return np.random.choice(greedy_choice) if len(greedy_choice) > 1 else greedy_choice[0]

    def e_pol(self, state_in):
        cnt = self.get_state_count(state_in)
        epsilon = self.explore / (self.explore + cnt-2)
        prob = epsilon / len(self.env.actions)
        probs = [prob for _ in self.env.actions]
        probs[self.greedy_pol(state_in)] += 1 - epsilon
        return np.random.choice(self.env.actions, p=probs)





