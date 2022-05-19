import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

from rlberry.envs.finite import FiniteMDP
from utls.mdp import SimpleMDP
from valueit.optimal_valueit import optimal_value_iteration

def c_func(a):
    return 2 * a

def O_func(K, a):
    if a > 0:
        return K + c_func(a)
    else:
        return 0

def h_func(x):
    return x

def f_func(x):
    return 8 * x

def F_func(M, D, u):
    Fu = 0
    if u < M:
        Fu = f_func(u) * np.sum(D[u:])
    for j in range(u):
        if j < M:
            Fu += f_func(j) * D[j]
    return Fu

# compute the optimal policy
def invent_opt(P, R, H, sigma, Sigma):
    S, A, _ = P.shape
    mdp = SimpleMDP(S, A, H, P, R)
    _, _, pi_star = optimal_value_iteration(mdp)
    np.save("./params/invent_20_pi_star", pi_star)

class Inventory(FiniteMDP):
    """
    Inventory environment.
    Parameters
    ----------
    
    """

    name = "Inventory"

    def __init__(self, M=6, fail_prob=0.1):
        self.M = M
        self.D = np.ones(6, dtype=float)/6  # uniform demand distribution
        self.K = 2
        self.initial_state_distribution = np.ones(6, dtype=float)/6

        
        # transition probabilities
        transitions = np.zeros((M, M, M))
        for ss in range(M):
            for aa in range(M):
                transitions[ss, aa, 0] = 1
        for ss in range(M):
            for aa in range(M - ss):
                for next_ss in range(M):
                    if (M >= ss + aa) and (ss + aa >= next_ss) and (next_ss > 0):
                        transitions[ss, aa, next_ss] = self.D[ss + aa - next_ss]
                    elif (M >= ss + aa) and (next_ss == 0):
                        transitions[ss, aa, next_ss] = np.sum(self.D[ss + aa:])
                    elif (M >= next_ss) and (next_ss > ss + aa):
                        transitions[ss, aa, next_ss] = 0
        rewards = np.zeros((M, M))
        for ss in range(M):
            for aa in range(M-1, M - ss - 1, -1):
                rewards[ss, aa] = 0
        for ss in range(M):
            for aa in range(M - ss):
                rewards[ss, aa] = - O_func(self.K, aa) - h_func(ss + aa) + F_func(M, self.D, ss + aa)
        
        for ss in range(M):
            rewards[ss] = rewards[ss] / np.sum(rewards[ss])             

        self.P = transitions
        self.R = rewards

        # init base classes
        FiniteMDP.__init__(self, rewards, transitions, initial_state_distribution=np.ones(M)/M)
        # reward_high = np.max(rewards)
        # reward_low = np.min(rewards)
        # self.reward_range = (reward_low, reward_high)
        self.reward_range = (-self.K * 8, self.K * 8)

    def reset(self):
        """
        Reset the environment to a default state.
        """
        if isinstance(self.initial_state_distribution, np.ndarray):
            self.state = self.rng.choice(
                self._states, p=self.initial_state_distribution
            )
        else:
            self.state = self.initial_state_distribution
        return self.state

    def set_initial_state_distribution(self, distribution):
        """
        Parameters
        ----------
        distribution : numpy.ndarray or int
            array of size (S,) containing the initial state distribution
            or an integer representing the initial/default state
        """
        self.initial_state_distribution = distribution
        self._check_init_distribution()

    def _check_init_distribution(self):
        if isinstance(self.initial_state_distribution, np.ndarray):
            assert abs(self.initial_state_distribution.sum() - 1.0) < 1e-15
        else:
            assert self.initial_state_distribution >= 0
            assert self.initial_state_distribution < self.S

    def reward_fn(self, state, action, next_state):
        """
        Reward function. Returns mean reward at (state, action) by default.
        Parameters
        ----------
        state : int
            current state
        action : int
            current action
        next_state :
            next state
        Returns:
            reward : float
        """
        # return - O_func(self.K, max(0, min(state + action, self.M - 1) - state)) - h_func(state) \
        #         + F_func(self.M, self.D, max(0, min(state + action, self.M - 1) - next_state))
        return - O_func(self.K, max(0, min(state + action, self.M - 1) - state)) - h_func(state) \
                + f_func(max(0, min(state + action, self.M - 1) - next_state))

    # def sample(self, state, action):
    #     """
    #     Sample a transition s' from P(s'|state, action).
    #     """
        
    #     prob = self.P[state, action, :]
    #     next_state = self.rng.choice(self._states, p=prob)
    #     reward = self.reward_fn(state, action, next_state)
    #     done = self.is_terminal(state)
    #     info = {}
    #     return next_state, reward, done, info

    def is_terminal(self, state):
        """
        Returns true if a state is terminal.
        """
        return False

    def step(self, action):
        assert action in self._actions, "Invalid action!"
        # take step
        demand = self.rng.choice(self.M, p=self.D)
        next_state = max(0, min(self.state + action, self.M - 1) - demand)
        reward = self.reward_fn(self.state, action, next_state)
        self.state = next_state
        # next_state, reward, done, info = self.sample(self.state, action)
        self.state = next_state
        done = self.is_terminal(self.state)
        info = {}
        return next_state, reward, done, info

if __name__ == '__main__':
    invent = Inventory(6)
    h = 20
    sigma = 2
    Sigma = 5
    invent_opt(invent.P, invent.R, h, 2, 5)