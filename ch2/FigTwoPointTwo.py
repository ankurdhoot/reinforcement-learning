import numpy as np
import random
import matplotlib.pyplot as plt

class BinaryBandit:
    """Simulates the binary bandit task intended to
    clarify the distinction between evaluative and instructive feedback
    as given in Section 2.4. In this specific instance, the number
    of actions is 2, although the class has been written to support
    more actions. Running BinaryBandit.generate_plots() will reproduce figure 2.3."""
    def __init__(self, num_plays, num_runs, epsilon, alpha, success_probs):
        # number of actions
        self.n = 2
        # number of plays in the game
        self.num_plays = num_plays
        # number of times to run simulation
        self.num_runs = num_runs
        # epsilon for action-value method
        self.epsilon = epsilon
        # alpha for L_rp and L_ri methods
        self.alpha = alpha
        # probability of success on each action (array of size n)
        self.success_probs = success_probs
        # estimates for epsilon greedy action values
        self.action_value_estimates = np.zeros(self.n)
        # number of times each epsilon greedy action has been played
        self.num_action_plays = np.zeros(self.n)
        # number of times an action was inferred to be correct
        self.supervised_tally = np.zeros(self.n)
        # action probability estimates for L_rp
        self.linear_reward_penalty_estimates = np.full(self.n, 1/self.n)
        # action probability estimates for L_ri
        self.linear_reward_inaction_estimates = np.full(self.n, 1/self.n)
        # counts whether the optimal action was played on the ith play, aggregated over all runs
        self.optimal_action_supervised = np.zeros(num_plays)
        self.optimal_action_epsilon_greedy = np.zeros(num_plays)
        self.optimal_action_L_rp = np.zeros(num_plays)
        self.optimal_action_L_ri = np.zeros(num_plays)

    def reset(self):
        """ Resets the arrays that aren't aggregated over runs. """
        self.action_value_estimates = np.zeros(self.n)
        self.num_action_plays = np.zeros(self.n)
        self.supervised_tally = np.zeros(self.n)
        self.linear_reward_penalty_estimates = np.full(self.n, 1 / self.n)
        self.linear_reward_inaction_estimates = np.full(self.n, 1 / self.n)

    def get_epsilon_greedy_action(self):
        """ Returns a random action with probability
        epsilon. Returns the greedy action according
        to the action_value_estimates otherwise. """
        if random.random() < self.epsilon:
            #play random action
            return random.randint(0, self.n - 1)
        else:
            #play greedy action
            m = max(self.action_value_estimates)
            max_indices = [i for i, j in enumerate(self.action_value_estimates) if j == m]
            return random.choice(max_indices)

    def get_supervised_action(self):
        """ Returns the action that has the most
        successes, similar to what a supervised
        algorithm might do."""
        m = max(self.supervised_tally)
        max_indices = [i for i, j in enumerate(self.supervised_tally) if j == m]
        return random.choice(max_indices)

    def get_L_rp_action(self):
        """ Selects an action according to the
        linear_reward_penalty_estimates probabilities."""
        return np.random.choice(self.n, size=1, p=self.linear_reward_penalty_estimates)[0]

    def get_L_ri_action(self):
        """ Selects an action according to the
         linear_reward_inaction_estimates probabilities."""
        return np.random.choice(self.n, size=1, p=self.linear_reward_inaction_estimates)[0]

    def get_action_reward(self, action):
        """ Returns success with probability
        success_probs[action], else failure."""
        success_prob = self.success_probs[action]
        if random.random() < success_prob:
            return 1
        return 0

    def update_epsilon_greedy_reward_estimates(self, action, reward):
        """Updates the play count for the given action.
        Updates the reward estimates for the given action
        using the incremental update implementation."""
        self.increment_action_count(action)
        old_estimate = self.action_value_estimates[action]
        new_estimate = old_estimate + 1 / self.num_action_plays[action] * (reward - old_estimate)
        self.action_value_estimates[action] = new_estimate

    def increment_action_count(self, action):
        """ Updates the number of times
        an action has been played."""
        self.num_action_plays[action] += 1

    def update_supervised_tally(self, action, reward):
        """ If the action was successful, increment
        it's success count. Else, distribute the success
        among all the other actions."""
        if reward == 1:
            self.supervised_tally[action] += 1
        else:
            for i in range(self.n):
                if i != action:
                    self.supervised_tally[i] += 1/(self.n - 1)

    def update_L_rp_estimates(self, action, reward):
        """ Updates the L-rp estimates on every action. """
        diff = reward - self.linear_reward_penalty_estimates[action]
        delta = self.alpha * diff
        for i in range(self.n):
            if i == action:
                self.linear_reward_penalty_estimates[i] += delta
            else:
                self.linear_reward_penalty_estimates[i] -= delta / (self.n - 1)

    def update_L_ri_estimates(self, action, reward):
        """ Updates the L_ri estimates only on successful actions. """
        diff = reward - self.linear_reward_inaction_estimates[action]
        delta = self.alpha * diff
        if reward == 1:
            for i in range(self.n):
                if i == action:
                    #floating point accuracy can sometimes make the prob > 1
                    self.linear_reward_inaction_estimates[i] = \
                        min(self.linear_reward_inaction_estimates[i] + delta, 1.0)
                else:
                    #floating point accuracy can sometimes make the prob < 0
                    self.linear_reward_inaction_estimates[i] = \
                        max(self.linear_reward_inaction_estimates[i] - delta / (self.n - 1), 0.0)

    def update_optimal_action_played(self, action, num_play, optimal_action_array):
        """Increment the num_play index of optimal_action_played
        if the action played is the best possible according to the
        action_values. Else, do nothing."""
        m = max(self.success_probs)
        max_indices = [i for i, j in enumerate(self.success_probs) if j == m]
        if action in max_indices:
            optimal_action_array[num_play] += 1

    def run_and_update_epsilon_greedy(self, num_play):
        action = self.get_epsilon_greedy_action()
        reward = self.get_action_reward(action)
        self.update_epsilon_greedy_reward_estimates(action, reward)
        self.update_optimal_action_played(action, num_play, self.optimal_action_epsilon_greedy)

    def run_and_update_supervised(self, num_play):
        action = self.get_supervised_action()
        reward = self.get_action_reward(action)
        self.update_supervised_tally(action, reward)
        self.update_optimal_action_played(action, num_play, self.optimal_action_supervised)

    def run_and_update_L_rp(self, num_play):
        action = self.get_L_rp_action()
        reward = self.get_action_reward(action)
        self.update_L_rp_estimates(action, reward)
        self.update_optimal_action_played(action, num_play, self.optimal_action_L_rp)

    def run_and_update_L_ri(self, num_play):
        action = self.get_L_ri_action()
        reward = self.get_action_reward(action)
        self.update_L_ri_estimates(action, reward)
        self.update_optimal_action_played(action, num_play, self.optimal_action_L_ri)


    def run_and_update(self, num_play):
        self.run_and_update_epsilon_greedy(num_play)
        self.run_and_update_supervised(num_play)
        self.run_and_update_L_rp(num_play)
        self.run_and_update_L_ri(num_play)

    def run_simulation(self):
        for run in range(self.num_runs):
            for play in range(self.num_plays):
                self.run_and_update(play)
            self.reset()
        optimal_action_epsilon_greedy = self.optimal_action_epsilon_greedy / self.num_runs
        optimal_action_supervised = self.optimal_action_supervised / self.num_runs
        optimal_action_L_rp = self.optimal_action_L_rp / self.num_runs
        optimal_action_L_ri = self.optimal_action_L_ri / self.num_runs
        return optimal_action_epsilon_greedy, optimal_action_supervised, optimal_action_L_rp, optimal_action_L_ri

    @staticmethod
    def generate_plots():
        epsilon = .1
        alpha = .1
        success_probs = [[.1, .2], [.8, .9]]
        trial_names = ['BANDIT A', 'BANDIT B']
        num_plays = 500
        num_runs = 2000
        f, ax = plt.subplots(len(trial_names))
        for trial in range(len(trial_names)):

            sim = BinaryBandit(num_plays=num_plays, num_runs=num_runs,
                               epsilon=epsilon, alpha=alpha, success_probs=success_probs[trial])
            optimal_greedy, optimal_supervised, optimal_L_rp, optimal_L_ri = sim.run_simulation()
            subplot = ax[trial]
            subplot.plot(optimal_greedy, label='action values')
            subplot.plot(optimal_supervised, label='supervised')
            subplot.plot(optimal_L_rp, label=r'$L_{R-P}$')
            subplot.plot(optimal_L_ri, label=r'$L_{R-I}$')
            subplot.set_title(trial_names[trial])
            subplot.set_ylabel('% Optimal Action')
            subplot.set_xlabel('Plays')
            subplot.legend(loc='lower right', shadow=True)

        plt.show()










