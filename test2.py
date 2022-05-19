import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from datetime import datetime
import pickle
from copy import deepcopy
from agents.ucbvi import UCBVIAgent
import os


import numpy as np
import pandas as pd
from envs.inventory import Inventory
from envs.frozenlake import FrozenLake
from utls.regret_writter import RandomAgent2, UCBVIAgent2, OptimalAgent
from rlberry.manager import AgentManager, plot_writer_data

if __name__ == '__main__':
    # Set environment
    env_ctor = FrozenLake # change for other environments
    i = 30
    k = 140
    t = 10
    prob = 0.2
    a = 1
    a2 = 2
    a3 = 3
    a4 = 4
    a5 = 5
    a6 = 6
    rand_seed = 5
    np.random.seed(rand_seed)
    env_kwargs = dict(slip_prob=prob, I = i, K = k, T = t, alpha = a, seed=rand_seed, map_name='map4')
    # env_kwargs2 = dict(slip_prob=prob, I = i, K = k, T = t, alpha = a2, seed=rand_seed)
    # env_kwargs3 = dict(slip_prob=prob, I = i, K = k, T = t, alpha = a3, seed=rand_seed)
    # env_kwargs4 = dict(slip_prob=prob, I = i, K = k, T = t, alpha = a4, seed=rand_seed)
    # env_kwargs5 = dict(slip_prob=prob, I = i, K = k, T = t, alpha = a5, seed=rand_seed)
    # env_kwargs6 = dict(slip_prob=prob, I = i, K = k, T = t, alpha = a6, seed=rand_seed)
    env1 = env_ctor(**env_kwargs)
    # env2 = env_ctor(**env_kwargs2)
    # env3 = env_ctor(**env_kwargs3)
    # env4 = env_ctor(**env_kwargs4)
    # env5 = env_ctor(**env_kwargs5)
    # env6 = env_ctor(**env_kwargs6)
    # env2 = env_ctor(**env_kwargs)
    # env3 = env_ctor(**env_kwargs)
    # env4 = env_ctor(**env_kwargs)
    # env5 = env_ctor(**env_kwargs)
    # env6 = env_ctor(**env_kwargs)
    
    fit_budget=200

    runs = 1

    ucb1 = np.zeros(fit_budget)
    ucb2 = np.zeros(fit_budget)
    
    optimal1 = OptimalAgent(env1)
    optimal1.fit(fit_budget)
    # optimal2 = OptimalAgent(env2)
    # optimal2.fit(fit_budget)
    # optimal3 = OptimalAgent(env3)
    # optimal3.fit(fit_budget)
    # optimal4 = OptimalAgent(env4)
    # optimal4.fit(fit_budget)
    # optimal5 = OptimalAgent(env5)
    # optimal5.fit(fit_budget)
    # optimal6 = OptimalAgent(env6)
    # optimal6.fit(fit_budget)
    
    # agent1
    ucbvi_params1 = {"env": env1, "gamma": 1.0, "horizon": 10, "bonus_scale_factor": 1.0}
    #baseline_params1 = {"env": env1, "gamma": 1.0, "horizon": 10, "bonus_scale_factor": 0.0, "bonus_type": "no bonus"}
    #baseline1 = UCBVIAgent2(**baseline_params1)
    #baseline1.fit(fit_budget)

    for i in range (runs):
        ucbvi1 = UCBVIAgent2(**ucbvi_params1)
        ucbvi1.fit(budget = fit_budget, seed = rand_seed)
        ucb1 += ucbvi1.value

    ucb1 = ucb1/runs
    
    # # agent2
    # ucbvi_params2 = {"env": env2, "gamma": 1.0, "horizon": 10, "bonus_scale_factor": 0.5}
    # #baseline_params2 = {"env": env2, "gamma": 1.0, "horizon": 10, "bonus_scale_factor": 0.0, "bonus_type": "no bonus"}
    # #baseline2 = UCBVIAgent2(**baseline_params2)
    # #baseline2.fit(fit_budget)

    # for i in range (runs):
    #     ucbvi2 = UCBVIAgent2(**ucbvi_params2)
    #     ucbvi2.fit(budget = fit_budget, seed = rand_seed)
    #     ucb2 += ucbvi2.value

    # ucb2 = ucb2/runs

    # # agent3
    
    # ucbvi_params3 = {"env": env3, "gamma": 1.0, "horizon": 10, "bonus_scale_factor": 0.3}
    # #baseline_params3 = {"env": env3, "gamma": 1.0, "horizon": 10, "bonus_scale_factor": 0.0}
    # ucbvi3 = UCBVIAgent2(**ucbvi_params3)
    # #baseline3 = UCBVIAgent2(**baseline_params3)
    # ucbvi3.fit(budget = fit_budget, seed = rand_seed)
    

    # # agent4
    # ucbvi_params4 = {"env": env4, "gamma": 1.0, "horizon": 10, "bonus_scale_factor": 0.1}
    # ucbvi4 = UCBVIAgent2(**ucbvi_params4)
    # ucbvi4.fit(budget = fit_budget, seed = rand_seed)

    # # agent5
    # ucbvi_params5 = {"env": env5, "gamma": 1.0, "horizon": 10, "bonus_scale_factor": 0.05}
    # ucbvi5 = UCBVIAgent2(**ucbvi_params5)
    # ucbvi5.fit(budget = fit_budget, seed = rand_seed)

    # # agent6
    # ucbvi_params6 = {"env": env6, "gamma": 1.0, "horizon": 10, "bonus_scale_factor": 0.01}
    # ucbvi6 = UCBVIAgent2(**ucbvi_params6)
    # ucbvi6.fit(budget = fit_budget, seed = rand_seed)

    regret1 = np.cumsum(optimal1.value - ucb1)
    #baseline_regret1 = np.cumsum(optimal1.value - baseline1.value)
    # regret2 = np.cumsum(optimal1.value - ucb2)
    #baseline_regret2 = np.cumsum(optimal2.value - baseline2.value)
    # regret3 = np.cumsum(optimal1.value - ucbvi3.value)
    #baseline_regret3 = np.cumsum(optimal3.value - baseline3.value)
    # regret4 = np.cumsum(optimal1.value - ucbvi4.value)
    # regret5 = np.cumsum(optimal1.value - ucbvi5.value)
    # regret6 = np.cumsum(optimal1.value - ucbvi6.value)

    '''
    episode = list(range(1,fit_budget+1))
    print(episode)
    plt.plot(episode, regret1, label = "alpha = 0.143")
    #plt.plot(episode, baseline_regret1, label = "b1")
    plt.plot(episode, regret2, label = "alpha = 0.286")
    #plt.plot(episode, baseline_regret2, label = "b2")
    plt.plot(episode, regret3, label = "alpha = 0.429")
    plt.plot(episode, regret4, label = "alpha = 0.571")
    plt.plot(episode, regret5, label = "alpha = 0.714")
    plt.plot(episode, regret6, label = "alpha = 0.857")
    plt.title("UCB Regret")
    plt.xlabel('Episodes')
    plt.ylabel('Culmulative Regret')
    plt.legend()
    plt.show()
    '''

    np.savetxt("./output/bonus_1_seed_{}.txt".format(rand_seed), regret1, delimiter=',')
    # np.savetxt("./output/bonus_5e-1_seed_{}.txt".format(rand_seed), regret2, delimiter=',')
    # np.savetxt("./output/bonus_3e-1_seed_{}.txt".format(rand_seed), regret3, delimiter=',')

    episode = list(range(1,fit_budget+1))
    print(episode)
    plt.plot(episode, regret1, label = "bonus_scale = 1.0")
    #plt.plot(episode, baseline_regret1, label = "b1")
    # plt.plot(episode, regret2, label = "bonus_scale = 0.5")
    #plt.plot(episode, baseline_regret2, label = "b2")
    # plt.plot(episode, regret3, label = "bonus_scale = 0.3")
    # plt.plot(episode, regret4, label = "bonus_scale = 0.1")
    # plt.plot(episode, regret5, label = "bonus_scale = 0.05")
    # plt.plot(episode, regret6, label = "bonus_scale = 0.01")
    plt.title("UCB Regret")
    plt.xlabel('Episodes')
    plt.ylabel('Culmulative Regret')
    plt.legend()
    plt.show()


    '''
    processed_df = read_writer_data(agent_manager, tag, preprocess_func)
    # add column with xtag, if given
    tag = "reward"

    data = processed_df[processed_df["tag"] == tag]

    xtag = "global_step"

    if data[xtag].notnull().sum() > 0:
        xx = xtag
        if data["global_step"].isna().sum() > 0:
            logger.warning(
                f"Plotting {tag} vs {xtag}, but {xtag} might be missing for some agents."
            )
    else:
        xx = data.index

    if ax is None:
        figure, ax = plt.subplots(1, 1)

    # PS: in the next release of seaborn, ci should be deprecated and replaced
    # with errorbar, which allows to specifies other types of confidence bars,
    # in particular quantiles.
    lineplot_kwargs = dict(
        x=xx, y="value", hue="name", style="name", data=data, ax=ax, ci="sd"
    )
    lineplot_kwargs.update(sns_kwargs)
    sns.lineplot(**lineplot_kwargs)
    ax.set_title(title)
    ax.set_ylabel(ylabel)

    if show:
        plt.show()

    if savefig_fname is not None:
        plt.gcf().savefig(savefig_fname)
    '''