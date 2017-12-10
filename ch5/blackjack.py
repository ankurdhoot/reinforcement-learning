import random
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Blackjack:
    """Simulates Example 5.1.
    To replicate the results in the book:
    b = Blackjack(20, 500000)
    b.simulate()
    b.generate_plots()
    """
    # TODO: replace all the numbers with appropriate variables
    def __init__(self, policy, num_episodes):
        # target sum of card values
        self.goal = 21
        # value below which the player hits
        self.policy = policy
        # state representation
        self.state = namedtuple('State', ['player_sum', 'usable_ace', 'dealer_card'])
        # {state : (estimate, number of times state has been seen)}
        self.state_value_estimates = dict()
        for player_sum in range(12, self.goal + 1):
            for dealer_card in range(1, 10 + 1):
                s1 = self.state(player_sum=player_sum, usable_ace=True, dealer_card=dealer_card)
                s2 = self.state(player_sum=player_sum, usable_ace=False, dealer_card=dealer_card)
                self.state_value_estimates[s1] = (0, 0)
                self.state_value_estimates[s2] = (0, 0)
        # number of episodes to run monte carlo simulation
        self.num_episodes = num_episodes

    def simulate(self):
        for episode in range(self.num_episodes):
            reward, states = self.play_episode()
            for state in states:
                current_estimate, num_visits = self.state_value_estimates[state]
                num_visits += 1
                new_estimate = current_estimate + 1 / num_visits * (reward - current_estimate)
                self.state_value_estimates[state] = (new_estimate, num_visits)


    def player_should_hit(self, state):
        if state.player_sum < self.policy:
            return True
        return False

    def draw_card(self):
        # 1 is ace, 10 is face card
        return random.randint(1, 10)

    def play_episode(self):
        states = []
        player_card_1 = self.draw_card()
        player_card_2 = self.draw_card()
        dealer_card_1 = self.draw_card()
        starting_state = self.create_starting_state(player_card_1, player_card_2, dealer_card_1)
        if self.is_natural(player_card_1, player_card_2):
            dealer_card_2 = self.draw_card()
            if self.is_natural(dealer_card_1, dealer_card_2):
                reward = 0
            else:
                reward = 1
            return reward, [starting_state]

        current_state = self.play_till_decision(starting_state)
        states.append(current_state)
        while self.player_should_hit(current_state):
            state_after_hit = self.hit_player(current_state)
            if state_after_hit:
                states.append(state_after_hit)
                current_state = state_after_hit
            else:
                return -1, states

        reward = self.play_dealer(current_state)
        return reward, states

    def create_starting_state(self, player_card_1, player_card_2, dealer_card_1):
        if player_card_1 == 1 or player_card_2 == 1:
            return self.state(player_sum=player_card_1 + player_card_2 + 10, usable_ace=True, dealer_card=dealer_card_1)
        else:
            return self.state(player_sum=player_card_1 + player_card_2, usable_ace=False, dealer_card=dealer_card_1)

    def play_till_decision(self, state):
        player_sum, usable_ace, dealer_card = state
        # TODO: make 12 a class variable instead of some magic number
        while player_sum < 12:
            card = self.draw_card()
            assert player_sum + card <= self.goal
            # is usable_ace == False redundant with player_sum < 11?
            if card == 1 and usable_ace == False and player_sum < 11:
                player_sum += 11
                usable_ace = True
            else:
                player_sum += card
        return self.state(player_sum=player_sum, usable_ace=usable_ace, dealer_card=dealer_card)


    def hit_player(self, state):
        player_sum, usable_ace, dealer_card = state
        assert 12 <= player_sum <= self.goal
        card = self.draw_card()
        # TODO: this logic can be restructured
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

    def is_natural(self, card_1, card_2):
        if card_1 == 1 or card_2 == 1:
            return card_1 + card_2 + 10 == self.goal
        return card_1 + card_2 == self.goal

    def generate_plot(self):
        x = np.arange(12, self.goal + 1)
        y = np.arange(1, 10 + 1)
        X, Y = np.meshgrid(x, y)
        z_usable_ace = np.array([self.state_value_estimates[self.state(x, True, y)][0] for x, y in zip(np.ravel(X), np.ravel(Y))])
        Z_usable_ace = z_usable_ace.reshape(X.shape)
        z_unusable_ace = np.array([self.state_value_estimates[self.state(x, False, y)][0] for x, y in zip(np.ravel(X), np.ravel(Y))])
        Z_unusable_ace = z_unusable_ace.reshape(X.shape)

        fig = plt.figure(1)
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(X, Y, Z_usable_ace)
        ax1.set_xlabel('player sum')
        ax1.set_ylabel('dealer card')
        ax1.set_title('Usable ace (%d episodes)' % self.num_episodes)
        ax1.view_init(elev=30, azim=-135)

        fig = plt.figure(2)
        ax2 = fig.add_subplot(111, projection='3d')
        ax2.scatter(X, Y, Z_unusable_ace)
        ax2.set_xlabel('player sum')
        ax2.set_ylabel('dealer card')
        ax2.set_title('No Usable ace (%d episodes)' % self.num_episodes)
        ax2.view_init(elev=30, azim=-135)