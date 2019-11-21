# %%
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# %%
class Easy21:

    def __init__(self, max_length=1000):
        self.max_length = max_length


    def reset(self):
        self.player_first_card_val = np.random.choice(10) + 1
        self.dealer_first_card_val = np.random.choice(10) + 1

        self.player_sum = self.player_first_card_val
        self.dealer_sum = self.dealer_first_card_val

        self.state = [self.dealer_first_card_val, self.player_sum]

        self.player_goes_bust = False
        self.dealer_goes_bust = False

        self.ret = 0
        self.terminal = False
        self.t = 0

        return self.state


    def step(self, action):
        # action 1: hit   0: stick
        # color: 1: black   -1: red
        r = 0

        if action == 1:
            self.player_card_val = np.random.choice(10) + 1
            self.player_card_col = np.random.choice([-1, 1], p=[1./3., 2./3.])

            self.player_sum += (self.player_card_val * self.player_card_col)
            self.player_goes_bust = self.check_go_bust(self.player_sum)

            if self.player_goes_bust == 1:
                r = -1
                self.terminal = True

        if not self.terminal and self.dealer_sum < 17:
            self.dealer_card_val = np.random.choice(10) + 1
            self.dealer_card_col = np.random.choice([-1, 1], p=[1./3., 2./3.])

            self.dealer_sum += (self.dealer_card_val * self.dealer_card_col)
            self.dealer_goes_bust = self.check_go_bust(self.dealer_sum)

            if self.dealer_goes_bust == 1:
                r = 1
                self.terminal = True

        self.t += 1
        self.ret += r

        if not self.terminal and self.t == self.max_length:
            self.terminal = True
            if self.player_sum > self.dealer_sum: r = 1
            elif self.player_sum < self.dealer_sum: r = -1

        if self.terminal: return 'Terminal', r, self.terminal
        else:
            self.state[1] = self.player_sum
            return self.state, r, self.terminal


    def check_go_bust(self, Sum):
        return bool(Sum > 21 or Sum < 1)

# %%
## Monte Carlo -- one episode
def monte_carlo(Q, Returns, count_state, count_state_action):
    appeared = np.zeros([10, 21, 2], dtype=int)

    actions = []
    rewards = []
    s = env.reset()
    states = [s]

    while True:
        action_greedy = Q[s[0]-1, s[1]-1, :].argmax()
        count_state[s[0]-1, s[1]-1] += 1
        epsilon = count_constant / float(count_constant + count_state[s[0]-1, s[1]-1])
        action = np.random.choice([action_greedy, 1 - action_greedy], p=[1. - epsilon/2., epsilon/2.])
        actions.append(action)

        s, r, term = env.step(action=action)
        rewards.append(r)

        if term: break
        else: states.append(s)

    for t in range(len(states)):
        
        ## ================== change here ================== ##
    
    
    
    
        ## ================================================= ##
    
    return Q, Returns, count_state, count_state_action

# %%
## Monte Carlo
Q_MC = np.zeros([10, 21, 2]) # Q(s, a)
Returns = np.zeros([10, 21, 2]) # empirical first-visit returns
count_state_action = np.zeros([10, 21, 2], dtype=int) # N(s, a)
count_state = np.zeros([10, 21], dtype=int) # N(s)
count_constant = 100

n_episodes = 20000
env = Easy21()

for i_epi in range(n_episodes):
    Q_MC, Returns, count_state, count_state_action = monte_carlo(Q_MC, Returns, count_state, count_state_action)

V_MC = Q_MC.max(axis=2)

# %%
## Monte Carlo -- plot
s1 = np.arange(10)+1
s2 = np.arange(21)+1
ss1, ss2 = np.meshgrid(s1, s2, indexing='ij')

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(ss1, ss2, V_MC, cmap=cm.coolwarm)

ax.set_xlabel("dealer's first card")
ax.set_ylabel("player's sum")
ax.set_zlabel("state value")
plt.yticks([1, 5, 10])
plt.yticks([1, 7, 14, 21])
fig.colorbar(surf, shrink=0.6)
fig.tight_layout()

plt.show()

# %%
## SARSA(lambda) function approximation -- coarse coding
def coarse_coding(s, a):
    v = np.zeros(3, dtype=int)
    for i in range(3):
        if (3 * i) <= s[0] <= (3 * (i+1)): v[i] = 1

    v_ = np.zeros(6, dtype=int)
    for i in range(6):
        if (3 * i) <= s[1] <= (3 * i + 5): v_[i] = 1
    v = np.append(v, v_)

    v_ = np.zeros(2, dtype=int)
    v_[a] = 1

    return np.append(v, v_)

# %%
## SARSA(lambda) function approximation -- one episode
def SARSA_lambda_func_approx(w, decay):
    epsilon = 0.05
    stepsize = 0.01

    s = env.reset()
    elig_trace = np.zeros(len(w))

    ## ================== change here ================== ##




    ## ================================================= ##
    
    return w

# %%
## SARSA(lambda) function approximation -- MSE vs. lambda
n_episodes = 1000
env = Easy21()

Decay = np.arange(0, 1.1, 0.1)
mse_Q_approx = np.zeros(len(Decay))

n_state_action = 10 * 21 * 2

feature = np.zeros([10, 21, 2, 11])
for i in range(10):
    for j in range(21):
        for a in range(2):
            feature[i, j, a, :] = coarse_coding([i, j], a)

for i_dec in range(len(Decay)):
    w = np.zeros(11)
    for i_epi in range(n_episodes):
        w = SARSA_lambda_func_approx(w, Decay[i_dec])

    Q_SARSA_approx = np.zeros([10, 21, 2])
    for i in range(10):
        for j in range(21):
            for a in range(2):
                Q_SARSA_approx[i, j, a] = np.dot(w, feature[i, j, a, :])

    mse_Q_approx[i_dec] = np.sum(np.square(Q_SARSA_approx - Q_MC)) / float(n_state_action)

print("The best lambda is:", Decay[mse_Q_approx.argmin()])

# %%
## SARSA(lambda) function approximation -- MSE vs. lambda plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.plot(Decay, mse_Q_approx, linewidth=2)

ax.set_xlabel("decay factor")
ax.set_ylabel("mse of Q")
fig.tight_layout()

plt.show()

# %%
## SARSA(lambda) function approximation -- learning curve when lambda = 0 or 1
mse_Q_approx_decay_0 = np.zeros(n_episodes)

w = np.zeros(11)
Q_SARSA_approx = np.zeros([10, 21, 2])
for i_epi in range(n_episodes):
    w = SARSA_lambda_func_approx(w, 0.)

    for i in range(10):
        for j in range(21):
            for a in range(2):
                Q_SARSA_approx[i, j, a] = np.dot(w, feature[i, j, a, :])

    mse_Q_approx_decay_0[i_epi] = np.sum(np.square(Q_SARSA_approx - Q_MC)) / float(n_state_action)


mse_Q_approx_decay_1 = np.zeros(n_episodes)

w = np.zeros(11)
Q_SARSA_approx = np.zeros([10, 21, 2])
for i_epi in range(n_episodes):
    w = SARSA_lambda_func_approx(w, 1.)

    for i in range(10):
        for j in range(21):
            for a in range(2):
                Q_SARSA_approx[i, j, a] = np.dot(w, feature[i, j, a, :])

    mse_Q_approx_decay_1[i_epi] = np.sum(np.square(Q_SARSA_approx - Q_MC)) / float(n_state_action)

# %%
## SARSA(lambda) function approximation -- learning curve when lambda = 0 or 1 plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.plot(np.arange(n_episodes), mse_Q_approx_decay_0, linewidth=2, c='olive', label='lambda = 0')
ax.plot(np.arange(n_episodes), mse_Q_approx_decay_1, linewidth=2, c='salmon', label='lambda = 1')

ax.set_xlabel("episode")
ax.set_ylabel("mse of Q")
ax.legend(loc='upper right', fontsize=10)

plt.show()