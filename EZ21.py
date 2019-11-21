import numpy as np


class EZ21:
    """
    Actions:
        STICK: 0
        HIT: 1
    Agents:
        DEALER: 0
        PLAYER: 1
    Colors:
        BLACK: 1
        RED: -1
    """
    actions = [0, 1]

    def __init__(self, max_length=1000):
        self.max_length = max_length
        self.game_time = 0
        self.state = [abs(EZ21.draw()), abs(EZ21.draw())]  # [Dealer score, Player Score]
        self.active_agent = 1

    def reset(self):
        self.game_time = 0
        self.state = [abs(EZ21.draw()), abs(EZ21.draw())]
        self.active_agent = 1
        return self.state

    def step(self, action):
        """
        :param int action: 0 for STICK, 1 for HIT
        :return:  game state, reward from step, True if game ended
        """
        rew = 0
        if action == 0:
            self.active_agent = 0

        self.state[self.active_agent] += EZ21.draw()

        # Reward if the dealer goes bust, punish if the player goes bust and return either way
        if self.state[self.active_agent] > 21 or self.state[self.active_agent] < 1:
            return 'Terminal', 1 - (self.active_agent * 2), True

        # If dealer is playing they might want to stick. No one is bust, so reward is dealt based on winner
        if self.active_agent == 0 and self.state[0] >= 17:
            return 'Terminal', self.reward_winner(), True

        # Timeout condition
        if self.game_time >= self.max_length:
            return 'Terminal', self.reward_winner(), True
        self.game_time += 1

        # If dealer is playing, recursively continue dealing cards until they win or bust.
        # We don't want to learn in this period
        if self.active_agent == 0:
            return self.step(action)

        # If player survived the round, output the updated state
        return self.state, rew, False

    @staticmethod
    def draw():
        agent_card_col = np.random.choice([-1, 1], p=[1. / 3., 2. / 3.])
        return agent_card_col * (np.random.choice(10) + 1)

    def reward_winner(self):
        return 0 if self.state[0] == self.state[1] else 1 - (self.state[0] > self.state[1]) * 2


