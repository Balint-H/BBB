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


class MonteCarlo:

    def __init__(self, env_in, Q_in=None, n_in=20000, explore_cnst=100):
        """
        dict Q[(state, action)] = [Q-value, number of updates]
        """
        self.env = env_in
        self.n = n_in
        self.Q = Q_in if Q_in is not None and isinstance(Q_in, dict) else dict()
        self.explore = explore_cnst

    def reset(self, Q_in=None):
        self.Q = Q_in if Q_in is not None and isinstance(Q_in, dict) else dict()

    def iter_opt(self):
        ret = list()
        for i in range(self.n):
            state = tuple(self.env.reset())
            trace = list()
            while True:
                act = self.e_pol(state)
                _state, rew, term = self.env.step(act) # Holding next state in buffer variable
                trace.append((state, act, rew))
                print("Trace is now: {0}".format(trace))
                if term:
                    ret.append(rew)
                    break
                state = tuple(_state)
            print("Beginning learning")
            for state, act, rew in trace:
                print("For state {0}, act {1}, Original Q value: {2}".format(state, act, self.get_Q(state, act)))
                self.Q[state, act] = [
                    self.get_Q(state, act) + (rew - self.get_Q(state, act)) / (self.get_state_count(state)-1),
                    self.get_invalpha(state, act) + 1]
                print("New Q value: {0}".format(self.get_Q(state, act)))
        return ret

    def get_Q(self, state_in, act_in):
        return self.Q[(state_in, act_in)][0] if (state_in, act_in) in self.Q else 0

    def get_state_count(self, state_in):
        return sum([self.get_invalpha(state_in, act) for act in self.env.actions])

    def get_invalpha(self, state_in, act_in):  # returns number of existing updates in current Q-value
        return self.Q[(state_in, act_in)][1] if (state_in, act_in) in self.Q else 1

    def get_V(self, state_in):
        return max([self.get_Q(state_in, act) for act in self.env.actions])

    def greedy_pol(self, state_in):
        print("Q-values for {0}: {1}".format(state_in, [self.get_Q(state_in, act) for act in self.env.actions]))
        greedy_choice = allmax([self.get_Q(state_in, act) for act in self.env.actions])
        ch = np.random.choice(greedy_choice) if len(greedy_choice)>1 else greedy_choice[0]
        print("Greedy chosen action is {0}".format(ch))
        return ch

    def e_pol(self, state_in):
        cnt = self.get_state_count(state_in)
        epsilon = self.explore / (self.explore + cnt-2)
        print("Epsilon is: {0}".format(epsilon))
        prob = epsilon / len(self.env.actions)
        probs = [prob for _ in self.env.actions]
        probs[self.greedy_pol(state_in)] += 1 - epsilon
        print("Choosing from {0} with probabilities {1}".format(self.env.actions, probs))
        ch = np.random.choice(self.env.actions, p=probs)
        print("Epsilon chosen {0}".format(ch))
        return ch





