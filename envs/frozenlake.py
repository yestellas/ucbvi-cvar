from tkinter import S
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

from utls.finite_mdp import FiniteMDP
from utls.mdp import SimpleMDP
from valueit.optimal_value_frozen_lake import optimal_value_iteration

import sys

np.set_printoptions(threshold=sys.maxsize)

# Mapping between directions and index number
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "map1": [
        "RRRRRRRRRR",
        "RHHHHHHHHR",
        "RI2IRRIIIR",
        "RIRIIIIIIR",
        "RGGGGGGH1R",
        "RGGGGGGH1R",
        "RIRIIIIIIR",
        "RI1IRRIIIR",
        "RHHHHHHHHR",
        "RRRRRRRRRR",
    ],
    "map2": [
        "RRRRR2RR",
        "RGGGGIHR",
        "RGRRRRRR",
        "RGGGGG1R",
        "RRRRRRRR",
    ],
    "map3": [
        "RR2R",
        "RGIR",
        "RGHR",
        "RRRR",
    ],
    "map4": [
        "RRRRRRRRR",
        "RHHHHHHHR",
        "RI2IRRIIR",
        "RIRIIIIIR",
        "RGGGGGH1R",
        "RIRIIIIIR",
        "RI1IRRIIR",
        "RHHHHHHHR",
        "RRRRRRRRR",
    ]
}

# compute the optimal policy
def invent_opt(P, R, H, sigma, Sigma):
    S, A, _ = P.shape
    mdp = SimpleMDP(S, A, H, P, R)
    _, _, pi_star = optimal_value_iteration(mdp)
    np.save("./params/frozenlake4_10_pi_star", pi_star)

class FrozenLake(FiniteMDP):
    """
    Inventory environment.
    Parameters
    ----------
    
    """

    name = "FrozenLake"

    def __init__(self, slip_prob, K, I, T, seed, alpha, desc=None, map_name="map4"):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.slip_prob = slip_prob
        self.seed = seed

        nA = 4
        unique, counts = np.unique(desc, return_counts=True)
        nS = (nrow * ncol - dict(zip(unique, counts))[b'R']) * 5

        self.T = T
        self.I = I
        self.Tr = {s : {a : [] for a in range(nA)} for s in range(nS)}
        self.P = np.zeros((nS, nA, nS))
        self.nP = np.zeros((nS, 2*I+1, nA, nS, 2*I+1))
        self.R = np.zeros((nS,nA))
        self.state = {s: [] for s in range(nS)}
        self.nS = self.getstatelength()
        self.nA = nA
        self.Rmax = 2
        self.K = K
        self.alpha = alpha

        Tr = self.getTransition(slip_prob)
        P = self.getTransitionProb(slip_prob)
        R = self.getReward(slip_prob)
        #nP = self.getTransitionNewProbFast(slip_prob)
        #print(nP[4,81,:,4,81])
        M = nS*(2*I+1)


        self.transitions = np.zeros((M, nA, M))
        self.transitions = self.getTransitionFast(slip_prob)

        self.rewards = np.zeros((M, nA))
        for s in range(self.nS):
            for i in range(-self.I, self.I+1):
                for a in range(4):
                    ss = s*(2*I+1)+i+I
                    self.rewards[ss, a] = self.R[s][a]

        self.P = self.transitions
        self.R = self.rewards

        #self.rewards = np.zeros((10, nA))

        # init base classes
        FiniteMDP.__init__(self, self.rewards, self.transitions, 0)


    def reset2(self):
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

    def reset(self):
        """
        Reset the environment to a default state.
        """

        self.state = 47*(self.I*2 +1) + self.I + self.alpha

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
        ss = state*(2*self.I+1)+i
        return self.rewards[state, action]

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

    def step2(self, action):
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

    def getNextI(self, r, i):
        for ni in range (-self.I, self.I+1):
            bi = self.T * self.Rmax / self.K * i
            bni = self.T * self.Rmax / self.K * ni
            bnni = self.T * self.Rmax / self.K * (ni + 1)
            if ((bnni - bi) > r) and ((bni - bi) <= r):
                return ni
        return ni

    def step(self, action):
        #np.random.seed(self.seed)
        assert action in self._actions, "Invalid action!"
        s = math.floor(self.state/(2*self.I+1))
        i = self.state%(2*self.I+1) - self.I
        #print(s, i, action)
        transitions = self.Tr[s][action]
        is_slip = np.random.rand()<(self.slip_prob) 
        p, s, r, d = transitions[1-is_slip] 
        ni = self.getNextI(r, i)
        self.state = s*(2*self.I+1)+ni+self.I
        s = self.state
        self.lastaction = action
        return (s, r, d, {"prob": p})

    def to_s(self, row, col, direction):
        a = self.desc[:row, :self.ncol]
        b = self.desc[row, :col]
        unique, counts = np.unique(a, return_counts=True)
        unique2, counts2 = np.unique(b, return_counts=True)
        c = 0
        d = 0
        if b'R' in dict(zip(unique, counts)):
            c = dict(zip(unique, counts))[b'R']
        if b'R' in dict(zip(unique2, counts2)):
            d = dict(zip(unique2, counts2))[b'R']
        if self.desc[row, col] in b'R':
            return -1
        else:
            return (row * self.ncol + col - c - d) * 5 + direction

    def getTransitionNewProbFast(self, slip_prob):
        for s in range(self.nS):
            for i in range(-self.I, self.I+1):
                for a in range(4):
                    prob, next_s, reward, terminal = self.Tr[s][a][0]
                    prob2, next_s2, reward2, terminal2 = self.Tr[s][a][1]
                    for ni in range (-self.I, self.I+1):
                        bi = self.T * self.Rmax / self.K * i
                        bni = self.T * self.Rmax / self.K * ni
                        bnni = self.T * self.Rmax / self.K * (ni + 1)
                        if ((bnni - bi) > self.R[s][a]) and ((bni - bi) <= self.R[s][a]):
                            self.nP[s][i][a][next_s][ni] = self.P[s][a][next_s] 
                            self.nP[s][i][a][next_s2][ni] = self.P[s][a][next_s2] 
        return self.nP

    def getTransitionFast(self, slip_prob):
        for s in range(self.nS):
            for i in range(-self.I, self.I+1):
                for a in range(4):
                    prob, next_s, reward, terminal = self.Tr[s][a][0]
                    prob2, next_s2, reward2, terminal2 = self.Tr[s][a][1]
                    for ni in range (-self.I, self.I+1):
                        bi = self.T * self.Rmax / self.K * i
                        bni = self.T * self.Rmax / self.K * ni
                        bnni = self.T * self.Rmax / self.K * (ni + 1)
                        if ((bnni - bi) > self.R[s][a]) and ((bni - bi) <= self.R[s][a]):
                            ss = s*(2*self.I+1)+(i+self.I)
                            next_ss1 = next_s*(2*self.I+1)+(ni+self.I)
                            next_ss2 = next_s2*(2*self.I+1)+(ni+self.I)
                            self.transitions[ss][a][next_ss1] = self.P[s][a][next_s] 
                            self.transitions[ss][a][next_ss2] = self.P[s][a][next_s2] 
        return self.transitions
    
    def getRewardsFast(self, slip_prob):
        for s in range(self.nS):
            for i in range(-self.I, self.I+1):
                for a in range(4):
                    prob, next_s, reward, terminal = self.Tr[s][a][0]
                    prob2, next_s2, reward2, terminal2 = self.Tr[s][a][1]
                    for ni in range (-self.I, self.I+1):
                        bi = self.T * self.Rmax / self.K * i
                        bni = self.T * self.Rmax / self.K * ni
                        bnni = self.T * self.Rmax / self.K * (ni + 1)
                        if ((bnni - bi) > self.R[s][a]) and ((bni - bi) <= self.R[s][a]):
                            self.nP[s][i][a][next_s][ni] = self.P[s][a][next_s] 
                            self.nP[s][i][a][next_s2][ni] = self.P[s][a][next_s2] 
        return self.nP

    def getReward(self, slip_prob):
        for s in range(self.nS):
            for a in range(4):
                for ns in range (self.nS):
                    row, col, direction = self.state[s][0]
                    letter = self.desc[row, col]
                    if letter in b'H12' :
                        self.R[s][a] = 0
                    else:
                        newrow, newcol, newdirection = self.nextstate2(row, col, direction, a, True)
                        newrow2, newcol2, newdirection2 = self.nextstate2(row, col, direction, a, False)
                        newletter = self.desc[newrow, newcol]
                        newletter2 = self.desc[newrow2, newcol2]
                        rew = 0
                        rew2 = 0
                        if newletter in b'12':
                            rew = float(newletter)
                        if newletter2 in b'12':
                            rew2  = float(newletter2)
                        self.R[s][a] = rew*slip_prob + rew2*(1-slip_prob)
        return self.R
    
    def getTransitionProb(self, slip_prob):
        for s in range(self.nS):
            for a in range(4):
                for ns in range (self.nS):
                    row, col, direction = self.state[s][0]
                    letter = self.desc[row, col]
                    if letter in b'12H':
                        if ns == s:
                            self.P[s][a][ns] = 1
                        else:
                            self.P[s][a][ns] = 0
                    else:
                        newrow, newcol, newdirection = self.nextstate2(row, col, direction, a, True)
                        newrow2, newcol2, newdirection2 = self.nextstate2(row, col, direction, a, False)
                        newstate = self.to_s(newrow, newcol, newdirection)
                        newstate2 = self.to_s(newrow2, newcol2, newdirection2)
                        if newstate == newstate2:
                            if ns == newstate:
                                self.P[s][a][ns] = 1
                            else:
                                self.P[s][a][ns] = 0
                        else:
                            if ns == newstate:
                                self.P[s][a][ns] = (slip_prob)
                            elif ns == newstate2:
                                self.P[s][a][ns] = (1-slip_prob)
                            else:
                                self.P[s][a][ns] = 0
        return self.P

    def getTransition(self, slip_prob):
        for row in range(self.nrow):
            for col in range(self.ncol):
                for direction in range(5):
                    s = self.to_s(row, col, direction)
                    if s != -1:
                        for a in range(4):
                            li = self.Tr[s][a]
                            letter = self.desc[row, col]
                            # is_slip = true
                            if letter in b'12EH':
                                li.append((slip_prob, s, 0, True))
                            else:
                                newrow, newcol, newdirection = self.nextstate2(row, col, direction, a, True)
                                newletter = self.desc[newrow, newcol]
                                newstate = self.to_s(newrow, newcol, newdirection)
                                done = bytes(newletter) in b'12EH'
                                rew = 0
                                if (newletter in b'12'):
                                    rew = float(newletter)
                                li.append((slip_prob, newstate, rew, done))

                            #is_slip = false
                            if letter in b'12EH':
                                li.append((1-slip_prob, s, 0, True))
                            else:
                                newrow, newcol, newdirection = self.nextstate2(row, col, direction, a, False)
                                newletter = self.desc[newrow, newcol]
                                newstate = self.to_s(newrow, newcol, newdirection)
                                done = bytes(newletter) in b'12EH'
                                rew = 0
                                if (newletter in b'12'):
                                    rew = float(newletter)
                                li.append((1-slip_prob, newstate, rew, done))
        return self.Tr

    def nextstate2(self, row, col, direction, a, is_slip):

        if direction != 4:
            a = direction

        if a == 0 and self.desc[row, max(col - 1, 0)] != b'R':  # left
            col = max(col - 1, 0)
            if (self.desc[row,col] == b'I' and is_slip
                    and self.desc[self.target(row, col, 0)] != b'R'):
                direction = 0
                prob = self.slip_prob
            else:
                direction = 4
                prob = 1 - self.slip_prob

        elif a == 1 and self.desc[min(row + 1, self.nrow - 1), col] != b'R':  # down
            row = min(row + 1, self.nrow - 1)
            if (self.desc[row,col] == b'I' and is_slip
                    and self.desc[self.target(row, col, 1)] != b'R'):
                direction = 1
                prob = self.slip_prob
            else:
                direction = 4
                prob = 1 - self.slip_prob

        elif a == 2 and self.desc[row, min(col + 1, self.ncol - 1)] != b'R':  # right
            col = min(col + 1, self.ncol - 1)
            if (self.desc[row,col] == b'I' and is_slip
                    and self.desc[self.target(row, col, 2)] != b'R'):
                direction = 2
                prob = self.slip_prob
            else:
                direction = 4
                prob = 1 - self.slip_prob

        elif a == 3 and self.desc[max(row - 1, 0), col] != b'R':  # up
            row = max(row - 1, 0)
            if (self.desc[row,col] == b'I' and is_slip
                    and self.desc[self.target(row, col, 3)] != b'R'):
                direction = 3
                prob = self.slip_prob
            else:
                direction = 4
                prob = 1 - self.slip_prob

        return row, col, direction

    def target(self, row, col, a):

        if a == 0:  # left
            col = max(col - 1, 0)

        elif a == 1:  # down
            row = min(row + 1, self.nrow - 1)

        elif a == 2:  # right
            col = min(col + 1, self.ncol - 1)

        elif a == 3:  # up
            row = max(row - 1, 0)

        return row, col

    def getstatelength(self):
        for row in range(self.nrow):
            for col in range(self.ncol):
                for direction in range(5):
                    s = self.to_s(row, col, direction)
                    if s != -1:
                        li = self.state[s]
                        li.append((row, col, direction))
        return len(self.state)
    
    

if __name__ == '__main__':
    #invent = FrozenLake(6)
    #h = 20
    #sigma = 2
    #Sigma = 5
    #invent_opt(invent.P, invent.R, h, 2, 5)
    i = 30
    k = 140
    t = 10
    prob = 0.2

    env = FrozenLake(slip_prob=prob, I = i, K = k, T = t, alpha=1, seed=5, map_name='map4')

    invent_opt(env.P, env.R, 10, 0, 0)

    print("done")
