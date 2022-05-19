import numpy as np

from agents.ucbvi import UCBVIAgent
from agents.random import RandomAgent
from rlberry.agents import Agent, ValueIterationAgent
from rlberry.envs import Chain
from rlberry.manager import AgentManager, evaluate_agents, plot_writer_data
from rlberry.wrappers import WriterWrapper
import os


class RandomAgent2(RandomAgent):
    name = "RandomAgent2"

    def __init__(self, env, **kwargs):
        RandomAgent.__init__(self, env, **kwargs)
        self.env = WriterWrapper(self.env, self.writer, write_scalar="reward")

class UCBVIAgent2(UCBVIAgent):
    name = "UCBVIAgent2"

    def __init__(self, env, **kwargs):
        UCBVIAgent.__init__(self, env, **kwargs)
        self.env = WriterWrapper(self.env, self.writer, write_scalar="reward")

# class VIAgent(ValueIterationAgent):
#     name = "ValueIterationAgent2"

#     def __init__(self, env, **kwargs):
#         ValueIterationAgent.__init__(self, env, **kwargs)
#         self.env = WriterWrapper(self.env, self.writer, write_scalar="reward")

class OptimalAgent(Agent):
    name = "OptimalAgent"

    def __init__(self, env, **kwargs):
        Agent.__init__(self, env, **kwargs)
        self.env = WriterWrapper(self.env, self.writer, write_scalar="reward")
        self.value = []

    def fit(self, budget=10, **kwargs):
        self.value = np.zeros(budget)

        # used for inventory environment only, change for other environments
        if (self.env.alpha == 1):
            pi_star = np.load("./params/frozenlake4_10_pi_star.npy")
            # pi_star = np.load("./params/frozenlake_10_1.npy")
        elif (self.env.alpha == 2):
            pi_star = np.load("./params/frozenlake_10_2.npy")
        elif (self.env.alpha == 3):
            pi_star = np.load("./params/frozenlake_10_3.npy")
        elif (self.env.alpha == 4):
            pi_star = np.load("./params/frozenlake_10_4.npy")
        elif (self.env.alpha == 5):
            pi_star = np.load("./params/frozenlake_10_5.npy")
        elif (self.env.alpha == 6):
            pi_star = np.load("./params/frozenlake_10_6.npy")
        else:
            pi_star = np.load("./params/frozenlake_10_pi_star_7.npy")

        H, _ = pi_star.shape
        observation = self.env.reset()
        n_sim = 200
        for ep in range(budget):
            value = 0
            for ii in range(n_sim):
                episode_rewards = 0
                state = self.env.reset()
                for h in range(H):
                    action =  pi_star[h, observation]
                    observation, reward, done, _ = self.env.step(action)
                    episode_rewards += reward
                #self.reward[ep] = episode_rewards
                value += episode_rewards
            value = value / n_sim
            self.value[ep] = value


    def policy(self, observation):
        # used for inventory environment only, change for other environments
        pi_star = np.load("./params/frozenlake_10_pi_star.npy")
        H, _ = pi_star.shape
        for j in range(4):
            if pi_star[j, observation, j] == 1:
                action = pi_star[j, observation, j]
        return action

    def eval(self, **kwargs):
        pass