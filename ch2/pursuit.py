from reinforcement_comparison import NArmedBandit, Simulator, EpsilonGreedyStationary, ReinforcementComparison
import numpy as np
import random
import matplotlib.pyplot as plt

class Pursuit(NArmedBandit):

    def __init__(self, n, num_plays, initial_action_value, beta):
        super().__init__(n, num_plays)
        # Set a high initial_action_value to encourage exploration at the start
        self.initial_action_value = initial_action_value
        # probability increment parameter
        self.beta = beta
        # running estimate of action values
        self.action_value_estimates = np.full(self.n, initial_action_value)
        # prob of taking each action
        self.action_probs = np.full(self.n, 1/self.n)

    def get_action_to_play(self):
        return np.random.choice(self.n, size=1, p=self.action_probs)


class PursuitStationary(Pursuit):
    """ Uses sample averages to estimate action values.
    Intended for stationary environments since equal weight
    is given to all samples. """

    def __init__(self, n, num_plays, initial_action_value, beta):
        super().__init__(n, num_plays, initial_action_value, beta)

    def reset(self):
        self.__init__(self.n, self.num_plays, self.initial_action_value, self.beta)

    def update_reward_estimates(self, action, reward):
        """ Updates action value estimates and
        action probabilities. """
        old_estimate = self.action_value_estimates[action]
        new_estimate = old_estimate + 1 / self.num_action_plays[action] * (reward - old_estimate)
        self.action_value_estimates[action] = new_estimate

        m = max(self.action_value_estimates)
        max_indices = [i for i, j in enumerate(self.action_value_estimates) if j == m]
        greedy_action = random.choice(max_indices)
        for action in range(self.n):
            if action == greedy_action:
                self.action_probs[action] += self.beta * (1 - self.action_probs[action])
            else:
                self.action_probs[action] += self.beta * (0 - self.action_probs[action])

def generate_plots():
    n = 10
    num_plays = 1000
    num_runs = 2000
    epsilon = .1
    initial_estimate = 0
    alpha = .1
    beta = .1
    initial_reward = 10  # encourage exploration
    epsilon_greedy_stationary = EpsilonGreedyStationary(n, num_plays, initial_estimate, epsilon)
    simulator = Simulator(num_plays, num_runs, epsilon_greedy_stationary)
    rewards, optimal_actions = simulator.simulate()
    f, (ax1, ax2) = plt.subplots(2)

    ax1.plot(rewards, label=r'$\epsilon$=' + str(epsilon) + r'$ \alpha=1/k$')
    ax2.plot(optimal_actions, label=r'$\epsilon$=' + str(epsilon) + r'$\alpha=1/k$')

    reinforcement_comparison = ReinforcementComparison(n, num_plays, initial_reward, alpha, beta)
    simulator = Simulator(num_plays, num_runs, reinforcement_comparison)
    rewards, optimal_actions = simulator.simulate()
    ax1.plot(rewards, label='reinforcement comparison')
    ax2.plot(optimal_actions, label='reinforcement comparison')

    beta = .01
    pursuit = PursuitStationary(n, num_plays, initial_estimate, beta)
    simulator = Simulator(num_plays, num_runs, pursuit)
    rewards, optimal_actions = simulator.simulate()
    ax1.plot(rewards, label='pursuit')
    ax2.plot(optimal_actions, label='pursuit')

    ax1.set_xlabel('Plays')
    ax1.set_ylabel('Average Reward')
    ax2.set_xlabel('Plays')
    ax2.set_ylabel('% Optimal Action')
    ax1.legend(loc='lower right', shadow=True)
    ax2.legend(loc='lower right', shadow=True)
    plt.show()