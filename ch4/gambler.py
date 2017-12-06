import numpy as np
import matplotlib.pyplot as plt


class Gambler:
    """ Replicates example 4.3: Gambler's Problem."""
    def __init__(self, p, goal):
        # probability of heads
        self.p = p
        # amount of money needed to win
        self.goal = goal
        # estimates of state value
        self.state_value_estimates = np.zeros(self.goal + 1, dtype=np.float64)
        # reward of 1 for reaching for reaching goal state
        self.state_value_estimates[goal] = 1
        # action is the amount of money to stake
        # state is the amount of capital the gambler currently has

    def get_actions(self, state):
        return [x for x in range(1, min(state, self.goal - state) + 1)]

    def state_action_value(self, state, action):
        assert 1 <= state < self.goal
        assert state - action >= 0
        assert state + action <= self.goal
        heads_value = self.p * self.state_value_estimates[state + action]
        tails_value = (1 - self.p) * self.state_value_estimates[state - action]
        return heads_value + tails_value

    def value_iteration(self):
        epsilon = .0000000001
        while True:
            delta = 0
            for state in range(1, self.goal):
                old_value = self.state_value_estimates[state]
                new_value = 0
                for action in self.get_actions(state):
                    new_value = max(new_value, self.state_action_value(state, action))

                self.state_value_estimates[state] = new_value
                delta = max(delta, abs(new_value - old_value))

            if delta < epsilon:
                break

    def get_greedy_policy(self):
        policy = [[0]]
        for state in range(1, self.goal):
            best_value = 0
            for action in self.get_actions(state):
                action_value = self.state_action_value(state, action)
                if action_value > best_value:
                    best_value = action_value
            best_actions = []
            for action in self.get_actions(state):
                action_value = self.state_action_value(state, action)
                if action_value == best_value:
                    best_actions.append(action)
            policy.append(best_actions)
        return policy

    def plot_policy(self):
        policy = self.get_greedy_policy()
        capital = [x for x in range(self.goal)]
        for xe, ye in zip(capital, policy):
            plt.scatter([xe] * len(ye), ye)
        plt.title("Optimal Policy for Gamblers Problem")
        plt.xlabel("Capital")
        plt.ylabel("Amount to stake (may not be unique)")
        plt.show()




