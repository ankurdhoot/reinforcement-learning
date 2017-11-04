import numpy as np
import matplotlib.pyplot as plt
import random

class NArmedTestbed:
    """ Simulates the n-armed bandit problem as given in Section 2.1.
    Running NArmedTestbed.generate_plots() will reproduce figure 2.1."""

    def __init__(self, n, num_plays, num_runs, epsilon):
        self.n = n
        self.num_plays = num_plays
        self.num_runs = num_runs
        self.epsilon = epsilon
        # actual action values
        self.action_values = np.random.normal(size=self.n)
        # running estimate of action values
        self.action_value_estimates = np.zeros(self.n)
        # number of times each action has been played
        self.num_action_plays = np.zeros(self.n)
        # reward returned on each play, aggregated over all runs
        self.rewards_per_play = np.zeros(self.num_plays)
        #counts whether the optimal action was played on the ith play, aggregated over all runs
        self.optimal_action_played = np.zeros(self.num_plays)

    def reset(self):
        """ Resets the arrays that aren't aggregated over runs. """
        self.action_values = np.random.normal(size=self.n)
        self.action_value_estimates = np.zeros(self.n)
        self.num_action_plays = np.zeros(self.n)


    def get_action_reward(self, action_num):
        """ Returns a random sample drawn from the Gaussian
        with mean = actual action value, deviation = 1."""
        mean = self.action_values[action_num]
        return np.random.normal(loc=mean, scale=1.0)

    def run_and_update(self, num_play):
        action = self.get_action_to_play(self.epsilon)
        reward = self.get_action_reward(action)
        self.update_reward_estimates(action, reward)
        self.update_optimal_action_played(action, num_play)
        self.update_reward_per_play(reward, num_play)

    def update_optimal_action_played(self, action, num_play):
        """Increment the num_play index of optimal_action_played
        if the action played is the best possible according to the
        action_values. Else, do nothing."""
        m = max(self.action_values)
        max_indices = [i for i, j in enumerate(self.action_values) if j == m]
        if action in max_indices:
            self.optimal_action_played[num_play] += 1

    def update_reward_estimates(self, action, reward):
        """Updates the play count for the given action.
        Updates the reward estimates for the given action
        using the incremental update implementation."""
        self.increment_action_count(action)
        old_estimate = self.action_value_estimates[action]
        new_estimate = old_estimate + 1 / self.num_action_plays[action] * (reward - old_estimate)
        self.action_value_estimates[action] = new_estimate

    def increment_action_count(self, action):
        self.num_action_plays[action] += 1

    def update_reward_per_play(self, reward, num_play):
        self.rewards_per_play[num_play] += reward

    def get_action_to_play(self, epsilon):
        """ Play a random action with probability epsilon.
        Else, play a greedy action according to the
        action_value_estimates. """
        if random.random() < epsilon:
            #play random action
            return random.randint(0, self.n - 1)
        else:
            #play greedy action
            m = max(self.action_value_estimates)
            max_indices = [i for i, j in enumerate(self.action_value_estimates) if j == m]
            return random.choice(max_indices)

    def run_simulation(self):
        """ Runs the n armed testbed simulation and returns
        the array of average reward per play along with the
        percentage of times the optimal action was selected
        during each play. """
        for run in range(self.num_runs):
            for play in range(self.num_plays):
                self.run_and_update(play)
            self.reset()
        average_rewards_per_play = self.rewards_per_play / self.num_runs
        optimal_action_percentage = self.optimal_action_played / self.num_runs
        return average_rewards_per_play, optimal_action_percentage

    @staticmethod
    def generate_plots():
        epsilons = [0, .01, .1]
        rewards = []
        actions = []
        num_actions = 10
        num_runs = 2000
        num_plays = 1000
        for epsilon in epsilons:
            sim = NArmedTestbed(n=num_actions, num_plays=num_plays, num_runs=num_runs, epsilon=epsilon)
            average_rewards_per_play, optimal_action_percentage = sim.run_simulation()
            rewards.append(average_rewards_per_play)
            actions.append(optimal_action_percentage)

        f, (ax1, ax2) = plt.subplots(2)
        ax1.set_xlabel('Plays')
        ax1.set_ylabel('Average Reward')
        ax2.set_xlabel('Plays')
        ax2.set_ylabel('% Optimal Action')
        for index, epsilon in enumerate(epsilons):
            ax1.plot(rewards[index], label=r'$\epsilon$=' + str(epsilon))
            ax2.plot(actions[index], label=r'$\epsilon$=' + str(epsilon))
        ax1.legend(loc='lower right', shadow=True)
        ax2.legend(loc='lower right', shadow=True)
        plt.show()



