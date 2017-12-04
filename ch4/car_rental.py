import numpy as np
from scipy.stats import poisson
from collections import defaultdict
import matplotlib.pyplot as plt
import math

class CarRentalEnvironment:
    """ Simulates Example 4.2: Jacks' Car Rental.
    Run c = CarRentalEnvironment(20, 20, 5, 2, 10, 3, 4, 3, 2, .9) to create
    an object that replicates the book example.
    c.generate_plots() will plot the policy after each policy iteration step.
    """
    # TODO: definitely need to clean this up
    def __init__(self, l1_max, l2_max, max_transfers, transfer_cost, profit_per_car,
                 l1_checkout_mean, l2_checkout_mean, l1_return_mean, l2_return_mean, gamma):
        # max cars allowed at location 1
        self.l1_max = l1_max
        # max cars allowed at location 2
        self.l2_max = l2_max
        # max number of cars that can be transferred in one night
        self.max_transfers = max_transfers
        # cost of transferring a car
        self.transfer_cost = transfer_cost
        # profit from renting out a car
        self.profit_per_car = profit_per_car
        # poisson mean for location 1 checkouts
        self.l1_checkout_mean = l1_checkout_mean
        # poisson mean for location 2 checkouts
        self.l2_checkout_mean = l2_checkout_mean
        # poisson mean for location 1 returns
        self.l1_return_mean = l1_return_mean
        # poisson mean for location 2 returns
        self.l2_return_mean = l2_return_mean
        # update parameter
        self.gamma = gamma
        # state value estimates
        self.state_value_estimates = dict()
        # each action is the number of cars to transfer
        self.actions = dict()
        for k in range(l1_max + 1):
            for j in range(l2_max + 1):
                state = (k, j)
                # initialize estimates to 0
                self.state_value_estimates[state] = 0
                # initial policy is to move no cars
                self.actions[state] = 0
        # TODO: make this non-arbitrary
        self.mean_factor = 4
        self.max_l1_checkouts = self.mean_factor * self.l1_checkout_mean
        self.max_l2_checkouts = self.mean_factor * self.l2_checkout_mean
        self.max_l1_returns = self.mean_factor * self.l1_return_mean
        self.max_l2_returns = self.mean_factor * self.l2_return_mean
        self.probs = defaultdict(list)
        means = {self.l1_checkout_mean, self.l2_checkout_mean, self.l1_return_mean, self.l2_return_mean}
        num_iterations = max(self.max_l1_checkouts, self.max_l2_checkouts, self.max_l1_returns, self.max_l2_returns)
        for mean in means:
            for k in range(num_iterations + 1):
                self.probs[mean].append(poisson.pmf(k, mean))

    # TODO: Make this function more modular
    def evaluate_policy(self):
        epsilon = .1
        new_state_values = dict()
        while True:
            print("New iteration")
            delta = 0
            for state, value in self.state_value_estimates.items():
                action = self.actions[state]
                value_update = self.evaluate_state_action(state, action)
                new_state_values[state] = value_update
                delta = max(delta, abs(value_update - value))
            self.state_value_estimates = new_state_values
            if delta < epsilon:
                break

    def evaluate_state_action(self, state, action):
        """ Use the current state_value_estimates to
        estimate Q(s,a). """
        value_update = 0
        l1_cars, l2_cars = state
        for num_checkouts_l1 in range(self.max_l1_checkouts):
            for num_checkouts_l2 in range(self.max_l2_checkouts):
                for num_returns_l1 in range(self.max_l1_returns):
                    for num_returns_l2 in range(self.max_l2_returns):
                        actual_checkouts_l1 = min(l1_cars, num_checkouts_l1)
                        actual_checkouts_l2 = min(l2_cars, num_checkouts_l2)
                        l1_remaining_cars = l1_cars - actual_checkouts_l1 + num_returns_l1
                        l2_remaining_cars = l2_cars - actual_checkouts_l2 + num_returns_l2
                        assert l1_remaining_cars >= 0 and l2_remaining_cars >= 0
                        reward_l1 = actual_checkouts_l1 * self.profit_per_car
                        reward_l2 = actual_checkouts_l2 * self.profit_per_car
                        if action >= 0:
                            cars_to_move = min(abs(action), l1_remaining_cars)
                        else:
                            cars_to_move = - min(abs(action), l2_remaining_cars)
                        l1_remaining_cars = min(l1_remaining_cars - cars_to_move, self.l1_max)
                        l2_remaining_cars = min(l2_remaining_cars + cars_to_move, self.l2_max)
                        assert l1_remaining_cars >= 0 and l2_remaining_cars >= 0
                        new_state = (l1_remaining_cars, l2_remaining_cars)
                        # charge for action instead?
                        # TODO: change to cars_to_move
                        total_reward = reward_l1 + reward_l2 - self.transfer_cost * abs(action)
                        # store probabilities in a dictionary for significant speed up
                        prob = self.probs[self.l1_checkout_mean][num_checkouts_l1] \
                               * self.probs[self.l2_checkout_mean][num_checkouts_l2] \
                               * self.probs[self.l1_return_mean][num_returns_l1] \
                               * self.probs[self.l2_return_mean][num_returns_l2]
                        value_update += prob * (total_reward + self.gamma * self.state_value_estimates[new_state])
        return value_update

    def improve_policy(self):
        stable_policy = True
        for state, value in self.state_value_estimates.items():
            old_action = self.actions[state]
            best_action = old_action
            best_value = value
            for action in range(-self.max_transfers, self.max_transfers + 1):
                new_value = self.evaluate_state_action(state, action)
                #print(state, action, new_value)
                if new_value > best_value:
                    best_value = new_value
                    best_action = action
            self.actions[state] = best_action
            if best_action != old_action:
                stable_policy = False
        return stable_policy

    def policy_iteration(self):
        while True:
            self.evaluate_policy()
            is_stable = self.improve_policy()
            if is_stable:
                break

    # TODO: rename this to something more descriptive
    def iterate_policy(self):
        self.evaluate_policy()
        return self.improve_policy()

    def generate_plots(self):
        num_iterations = 0
        policies_over_time = [self.actions.copy()]
        while not self.iterate_policy():
            num_iterations += 1
            policies_over_time.append(self.actions.copy())

        for plot_num in range(num_iterations + 1):
            self.plot_actions(policies_over_time[plot_num], plot_num)

    def plot_actions(self, actions, plot_num):
        f = plt.figure(plot_num)
        levels = [level for level in range(-self.max_transfers, self.max_transfers + 1)]
        x = np.arange(0, self.l2_max + 1)
        y = np.arange(0, self.l1_max + 1)
        X, Y = np.meshgrid(x, y)
        axis1, axis2 = X.shape
        Z = np.zeros(shape=(axis1, axis2))
        for i in range(axis1):
            for j in range(axis2):
                Z[i][j] = actions[(i, j)]
        CS = plt.contour(X, Y, Z, levels=levels)
        plt.clabel(CS, inline=1, fontsize=10)
        plt.xlabel('#Cars at second location')
        plt.ylabel('#Cars at first location')
        plt.title(r'$\pi_{%d}$' % plot_num, fontsize=24)
        plt.show()
