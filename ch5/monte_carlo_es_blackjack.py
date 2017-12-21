# Need to store Q(s,a) instead of V(s)
# Need to create policy based on states
# Use exploring starts - start with all state action pairs
# Need to generate the starting states at random
# Use np.argmax to select action
# generate three times as many 10 cards

import random
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt

class MCBlackjack:
    """ Intended to replicate Figure 5.3.
    b = MCBlackjack(500000)
    b.simulate()
    b.generate_plots()
    """
    def __init__(self, num_episodes):
        self.goal = 21
        self.num_episodes = num_episodes
        self.initial_estimate = 0
        self.state = namedtuple('State', ['player_sum', 'usable_ace', 'dealer_card'])
        self.state_action = namedtuple('State_Action', ['state', 'action'])
        # player hits if player_sum >= initial_policy
        self.initial_policy = 20
        # mapping from state to action
        # True = hit, False = stick
        # {state : boolean}
        self.policy = dict()
        # {state_action : (estimate, number of times state-action has been seen)}
        self.state_action_estimates = dict()
        for player_sum in range(12, self.goal + 1):
            for dealer_card in range(1, 10 + 1):
                s1 = self.state(player_sum=player_sum, usable_ace=True, dealer_card=dealer_card)
                s2 = self.state(player_sum=player_sum, usable_ace=False, dealer_card=dealer_card)

                s1_hit = self.state_action(state=s1, action=True)
                s1_stick = self.state_action(state=s1, action=False)

                s2_hit = self.state_action(state=s2, action=True)
                s2_stick = self.state_action(state=s2, action=False)

                self.state_action_estimates[s1_hit] = (self.initial_estimate, 0)
                self.state_action_estimates[s1_stick] = (self.initial_estimate, 0)
                self.state_action_estimates[s2_hit] = (self.initial_estimate, 0)
                self.state_action_estimates[s2_stick] = (self.initial_estimate, 0)

                if player_sum >= self.initial_policy:
                    self.policy[s1] = False
                    self.policy[s2] = False
                else:
                    self.policy[s1] = True
                    self.policy[s2] = True

    def draw_card(self):
        # 1 is ace, 10 is face card
        card = random.randint(1, 13)
        return min(card, 10)

    def play_episode(self):
        state_actions = []
        current_state = self.create_starting_state()
        # generate first action at random
        action = random.choice([True, False])
        if action:
            state_actions.append(self.state_action(current_state, action=True))
            state_after_hit = self.hit_player(current_state)
            if not state_after_hit:
                return -1, state_actions
            current_state = state_after_hit
        else:
            reward = self.play_dealer(current_state)
            state_actions.append(self.state_action(current_state, action=False))
            return reward, state_actions

        while self.player_should_hit(current_state):
            state_actions.append(self.state_action(current_state, action=True))
            state_after_hit = self.hit_player(current_state)
            if state_after_hit:
                current_state = state_after_hit
            else:
                return -1, state_actions
        state_actions.append(self.state_action(current_state, action=False))

        reward = self.play_dealer(current_state)
        return reward, state_actions

    def play_till_decision(self, state):
        player_sum, usable_ace, dealer_card = state
        while player_sum < 12:
            card = self.draw_card()
            assert player_sum + card <= self.goal
            # is usable_ace == False redundant with player_sum < 11?
            if card == 1 and not usable_ace and player_sum < 11:
                player_sum += 11
                usable_ace = True
            else:
                player_sum += card
        return self.state(player_sum=player_sum, usable_ace=usable_ace, dealer_card=dealer_card)

    def player_should_hit(self, state):
        """ Uses the policy to determine whether
        the player should hit. True = hit, False = stick."""
        return self.policy[state]

    def hit_player(self, state):
        player_sum, usable_ace, dealer_card = state
        assert 12 <= player_sum <= self.goal
        card = self.draw_card()
        if player_sum + card <= self.goal:
            player_sum += card
        elif usable_ace:
            # count the previously usable_ace as a 1 instead of 11
            usable_ace = False
            player_sum += card - 10
        else:
            # the player went bust
            return None
        return self.state(player_sum=player_sum, usable_ace=usable_ace, dealer_card=dealer_card)

    def is_natural(self, card_1, card_2):
        if card_1 == 1 or card_2 == 1:
            return card_1 + card_2 + 10 == self.goal
        return card_1 + card_2 == self.goal

    def create_starting_state(self):
        player_sum = random.randint(12, self.goal)
        dealer_card = random.randint(1, 10)
        usable_ace = random.choice([True, False])
        return self.state(player_sum=player_sum, dealer_card=dealer_card, usable_ace=usable_ace)

    def play_dealer(self, state):
        player_sum, usable_ace, dealer_card = state
        dealer_has_usable_ace = False
        dealer_sum = dealer_card
        if dealer_card == 1:
            dealer_has_usable_ace = True
            dealer_sum = 11

        while dealer_sum < 17:
            card = self.draw_card()
            if card == 1 and dealer_sum + 11 <= self.goal:
                dealer_has_usable_ace = True
                dealer_sum += 11
            elif dealer_sum + card <= self.goal:
                dealer_sum += card
            else:
                # card puts dealer over the top
                if dealer_has_usable_ace:
                    dealer_sum += card - 10
                    dealer_has_usable_ace = False
                else:
                    # dealer went bust
                    return 1
        assert dealer_sum <= self.goal
        assert player_sum <= self.goal
        if dealer_sum > player_sum:
            return -1
        elif dealer_sum == player_sum:
            return 0
        else:
            return 1

    def simulate(self):
        for episode in range(self.num_episodes):
            reward, state_actions = self.play_episode()
            for state_action in state_actions:
                current_estimate, num_visits = self.state_action_estimates[state_action]
                num_visits += 1
                new_estimate = current_estimate + 1 / num_visits * (reward - current_estimate)
                self.state_action_estimates[state_action] = (new_estimate, num_visits)
            # update policy
            for state in self.policy:
                state_hit = self.state_action(state, action=True)
                state_stick = self.state_action(state, action=False)
                state_hit_value = self.state_action_estimates[state_hit]
                state_stick_value = self.state_action_estimates[state_stick]
                if state_hit_value == state_stick_value:
                    self.policy[state] = random.choice([True, False])
                elif state_hit_value > state_stick_value:
                    self.policy[state] = True
                else:
                    self.policy[state] = False


    def generate_plots(self):
        x = np.arange(12, self.goal + 1)
        y = np.arange(1, 10 + 1)
        X, Y = np.meshgrid(x, y)
        z_usable_ace = np.array([self.policy[self.state(x, True, y)] for x, y in zip(np.ravel(X), np.ravel(Y))])
        Z_usable_ace = z_usable_ace.reshape(X.shape)

        z_unusable_ace = np.array([self.policy[self.state(x, False, y)] for x, y in zip(np.ravel(X), np.ravel(Y))])
        Z_unusable_ace = z_unusable_ace.reshape(X.shape)

        c_usable = np.zeros((10, 10))
        c_unusable = np.zeros((10, 10))

        for x in range(1, 11):
            for y in range(12, self.goal + 1):
                c_usable[abs(y - self.goal), x - 1] = self.policy[self.state(y, True, x)]
                c_unusable[abs(y - self.goal), x - 1] = self.policy[self.state(y, False, x)]

        plt.figure()
        plt.imshow(c_usable, cmap='Paired', extent=[1, 10, 12, self.goal], interpolation='nearest')
        plt.xlabel('Dealer Card')
        plt.ylabel('Player Sum')
        plt.title('Usable Ace')

        plt.figure()
        plt.imshow(c_unusable, cmap='Paired', extent=[1, 10, 12, self.goal], interpolation='nearest')
        plt.xlabel('Dealer Card')
        plt.ylabel('Player Sum')
        plt.title('No Usable Ace')