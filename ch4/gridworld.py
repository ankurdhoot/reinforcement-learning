from enum import Enum


class GridEnvironment:
    """ Replicates the gridworld example in 4.2.
    All actions are given equal probability in any state.
    The reward is always -1 on a transition.
    To exactly replicate the 4.2 grid :
    grid = GridEnvironment(4, 4, {(0,0), (3,3)}, 1)
    grid.evaluate_policy() will evaluate the equiprobable policy.
    """
    def __init__(self, m, n, terminal_states, gamma):
        self.m = m
        self.n = n
        self.terminal_states = terminal_states
        self.gamma = gamma
        self.state_value_estimates = dict()
        for x in range(m):
            for y in range(n):
                self.state_value_estimates[(x, y)] = 0

    def get_reward(self, s, a, s_prime):
        if s in self.terminal_states:
            return 0
        return -1  # return 0 for s = terminal state or throw exception?

    def get_action_prob(self, s, a):
        if s in self.terminal_states:
            return 0
        return 1/len(Action)

    def get_transition_prob(self, s, a, s_prime):
        if s in self.terminal_states:
            return 0
        action_result = tuple(map(sum, zip(s, a)))
        possible_resultants = self.get_possible_resultant_states(s)
        if action_result == s_prime and action_result in possible_resultants:
            return 1
        elif s == s_prime and action_result not in possible_resultants:
            return 1
        else:
            return 0

    def is_valid_state(self, state):
        x, y = state
        return 0 <= x <= self.m - 1 and 0 <= y <= self.n - 1

    def get_possible_resultant_states(self, state):
        if state in self.terminal_states:
            return set()
        resultants = set()
        for action in Action:
            resultant_state = tuple(map(sum, zip(state, action.value)))
            if self.is_valid_state(resultant_state):
                resultants.add(resultant_state)
            else:
                resultants.add(state)
        return resultants

    def evaluate_policy(self):
        epsilon = .00000001
        while True:
            delta = 0
            current = self.state_value_estimates.copy()
            for state, value in self.state_value_estimates.items():
                value_update = 0
                for action in Action:
                    action_prob = self.get_action_prob(state, action.value)
                    for s_prime in self.get_possible_resultant_states(state):
                        transition_prob = self.get_transition_prob(state, action.value, s_prime)
                        value_update += action_prob * transition_prob * \
                                        (self.get_reward(state, action.value, s_prime) + self.gamma * current[s_prime])
                self.state_value_estimates[state] = value_update
                delta = max(delta, abs(value - value_update))
            if delta < epsilon:
                break

    def print_state_values(self):
        for x in range(self.m):
            row_string = ''
            for y in range(self.n):
                row_string += "%.1f" % self.state_value_estimates[(x, y)] + ' '
            print(row_string)

class Action(Enum):
    North = (-1, 0)
    South = (1, 0)
    East = (0, 1)
    West = (0, -1)

