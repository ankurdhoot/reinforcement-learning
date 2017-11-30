import numpy as np
import matplotlib.pyplot as plt
import random
import math

# TODO: Split all these classes into their own files
class NArmedBandit:
    """ Simulates the n-armed bandit problem as given in Section 2.1.
    Running NArmedTestbed.generate_plots() will reproduce figure 2.1."""

    def __init__(self, n, num_plays):
        self.n = n
        self.num_plays = num_plays
        # the number of plays played
        self.play_num = 0
        # actual action values
        self.action_values = np.random.normal(size=self.n)
        # reward returned on each play, aggregated over all runs
        self.rewards_per_play = np.zeros(self.num_plays)
        # counts whether the optimal action was played on the ith play
        self.optimal_action_played = np.zeros(self.num_plays)
        # number of times each action has been played
        self.num_action_plays = np.zeros(self.n)

    def play_action(self, action):
        """ Increments the action count for this action.
        Updates the optimal_action_played and rewards_per_play arrays.
        Returns the reward for the action."""
        self.increment_action_count(action)
        reward = self.get_reward(action)
        self.update_reward_per_play(reward)
        self.update_optimal_action_played(action)
        self.increment_play_num()
        return reward

    def get_reward(self, action):
        return np.random.normal(loc=self.action_values[action], scale=1.0)

    def increment_action_count(self, action):
        self.num_action_plays[action] += 1

    def increment_play_num(self):
        self.play_num += 1

    def update_optimal_action_played(self, action):
        """Increment the num_play index of optimal_action_played
        if the action played is the best possible according to the
        action_values. Else, do nothing."""
        m = max(self.action_values)
        max_indices = [i for i, j in enumerate(self.action_values) if j == m]
        if action in max_indices:
            self.optimal_action_played[self.play_num] = 1

    def update_reward_per_play(self, reward):
        self.rewards_per_play[self.play_num] = reward

    # TODO: Change to use abc module
    # TODO: Maybe rename to change_environment_estimates
    # (parameters other than the reward may need to be updated)?
    def update_reward_estimates(self, action, reward):
        """ Updates the reward estimates for the given action
        using the incremental update implementation."""
        raise NotImplementedError("Must implement update_reward_estimates.")

    def get_action_to_play(self):
        raise NotImplementedError("Must implement get_action_to_play.")

    # TODO: Make reset() an abstract method?

    def run_simulation(self):
        """Plays all the plays and returns the
        rewards and optimal actions arrays."""
        for play in range(self.num_plays):
            action = self.get_action_to_play()
            reward = self.play_action(action)
            self.update_reward_estimates(action, reward)
        return self.rewards_per_play, self.optimal_action_played


# TODO: Create an interface that the learning algorithm classes must implement
class EpsilonGreedy(NArmedBandit):
    def __init__(self, n, num_plays, initial_estimate, epsilon):
        super().__init__(n, num_plays)
        # probability of choosing a random action
        self.epsilon = epsilon
        # the starting value of each action
        self.initial_estimate = initial_estimate
        # running estimate of action values
        self.action_value_estimates = np.full(self.n, self.initial_estimate)

    def get_action_to_play(self):
        """ Play a random action with probability epsilon.
        Else, play a greedy action according to the
        action_value_estimates. """
        if random.random() < self.epsilon:
            #play random action
            return random.randint(0, self.n - 1)
        else:
            #play greedy action
            m = max(self.action_value_estimates)
            max_indices = [i for i, j in enumerate(self.action_value_estimates) if j == m]
            return random.choice(max_indices)


class EpsilonGreedyStationary(EpsilonGreedy):
    """ Uses the average update rule for reward estimates.
    Intended for use when the action values are stationary. """
    def __init__(self, n, num_plays, initial_estimate, epsilon):
        super().__init__(n, num_plays, initial_estimate, epsilon)

    def reset(self):
        self.__init__(self.n, self.num_plays, self.initial_estimate, self.epsilon)

    def update_reward_estimates(self, action, reward):
        old_estimate = self.action_value_estimates[action]
        new_estimate = old_estimate + 1 / self.num_action_plays[action] * (reward - old_estimate)
        self.action_value_estimates[action] = new_estimate


class EpsilonGreedyNonStationary(EpsilonGreedy):
    """ Uses a constant step size parameter, alpha, for reward estimates.
    Desirable for nonstationary environments. """

    def __init__(self, n, num_plays, initial_estimate, epsilon, alpha):
        super().__init__(n, num_plays, initial_estimate, epsilon)
        self.alpha = alpha

    def reset(self):
        self.__init__(self.n, self.num_plays, self.initial_estimate, self.epsilon, self.alpha)

    def update_reward_estimates(self, action, reward):
        old_estimate = self.action_value_estimates[action]
        new_estimate = old_estimate + self.alpha * (reward - old_estimate)
        self.action_value_estimates[action] = new_estimate


class ReinforcementComparison(NArmedBandit):
    def __init__(self, n, num_plays, initial_reward, alpha, beta):
        super().__init__(n, num_plays)
        # the starting value of the reference reward
        self.initial_reward = initial_reward
        # the reference reward to be updated
        self.reference_reward = initial_reward
        # step size for reward update
        self.alpha = alpha
        # step size for preferences
        self.beta = beta
        # running estimate of action preferences
        self.action_preference_estimates = np.zeros(self.n)

    def reset(self):
        self.__init__(self.n, self.num_plays, self.initial_reward, self.alpha, self.beta)

    def get_action_to_play(self):
        """ Select the action according to the soft-max probabilities."""
        probs = np.zeros(self.n)
        denom = 0
        for action in range(self.n):
            denom += math.exp(self.action_preference_estimates[action])
        for action in range(self.n):
            probs[action] = math.exp(self.action_preference_estimates[action]) / denom
        return np.random.choice(self.n, size=1, p=probs)

    def update_reward_estimates(self, action, reward):
        """ Update the action preference estimates using using the constant update
        parameter, beta. Update the reference reward using the constant update
        parameter, alpha."""
        self.action_preference_estimates[action] += self.beta * (reward - self.reference_reward)
        self.reference_reward += self.alpha * (reward - self.reference_reward)

class Simulator:
    def __init__(self, num_plays, num_runs, reinforcement_method):
        self.num_plays = num_plays
        self.num_runs = num_runs
        self.reinforcement_method = reinforcement_method

    def simulate(self):
        """ Runs the reinforcement methods for num_runs.
        Returns the average rewards, and optimal action percentage arrays."""
        rewards = np.zeros(self.num_plays)
        optimal_actions = np.zeros(self.num_plays)
        for run in range(self.num_runs):
            run_rewards, run_optimal_actions = self.reinforcement_method.run_simulation()
            rewards += run_rewards
            optimal_actions += run_optimal_actions
            self.reinforcement_method.reset()
        return rewards/self.num_runs, optimal_actions/self.num_runs

def generate_plots():
    n = 10
    num_plays = 1000
    num_runs = 2000
    epsilon = .1
    initial_estimate = 0
    alpha = .1
    beta = .1
    initial_reward = 10  #encourage exploration
    epsilon_greedy_stationary = EpsilonGreedyStationary(n, num_plays, initial_estimate, epsilon)
    simulator = Simulator(num_plays, num_runs, epsilon_greedy_stationary)
    rewards, optimal_actions = simulator.simulate()
    f, (ax1, ax2) = plt.subplots(2)

    ax1.plot(rewards, label=r'$\epsilon$=' + str(epsilon) + r'$ \alpha=1/k$')
    ax2.plot(optimal_actions, label=r'$\epsilon$=' + str(epsilon) + r'$\alpha=1/k$')

    epsilon_greedy_nonstationary = EpsilonGreedyNonStationary(n, num_plays, initial_estimate, epsilon, alpha)
    simulator = Simulator(num_plays, num_runs, epsilon_greedy_nonstationary)
    rewards, optimal_actions = simulator.simulate()
    ax1.plot(rewards, label=r'$\epsilon$=' + str(epsilon) + r'$ \alpha=$' + str(alpha))
    ax2.plot(optimal_actions, label=r'$\epsilon$=' + str(epsilon) + r'$\alpha=$' + str(alpha))

    reinforcement_comparison = ReinforcementComparison(n, num_plays, initial_reward, alpha, beta)
    simulator = Simulator(num_plays, num_runs, reinforcement_comparison)
    rewards, optimal_actions = simulator.simulate()
    ax1.plot(rewards, label='reinforcement comparison')
    ax2.plot(optimal_actions, label='reinforcement comparison')

    ax1.set_xlabel('Plays')
    ax1.set_ylabel('Average Reward')
    ax2.set_xlabel('Plays')
    ax2.set_ylabel('% Optimal Action')
    ax1.legend(loc='lower right', shadow=True)
    ax2.legend(loc='lower right', shadow=True)
    plt.show()