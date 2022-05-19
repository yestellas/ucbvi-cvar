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

    env_kwargs4 = dict(slip_prob=prob, I = i, K = k, T = t, alpha = a4, seed=rand_seed)

    env4 = env_ctor(**env_kwargs4)

    
    fit_budget=120

    runs = 1

    ucb1 = np.zeros(fit_budget)
    ucb2 = np.zeros(fit_budget)

    optimal4 = OptimalAgent(env4)
    optimal4.fit(fit_budget)

    # agent4
    ucbvi_params4 = {"env": env4, "gamma": 1.0, "horizon": 10, "bonus_scale_factor": 1.0}
    ucbvi4 = UCBVIAgent2(**ucbvi_params4)
    ucbvi4.fit(budget = fit_budget, seed = rand_seed)

    ucbvi_params42 = {"env": env4, "gamma": 1.0, "horizon": 10, "bonus_scale_factor": 0.5}
    ucbvi42 = UCBVIAgent2(**ucbvi_params4)
    ucbvi4.fit(budget = fit_budget, seed = rand_seed)

    ucbvi_params43 = {"env": env4, "gamma": 1.0, "horizon": 10, "bonus_scale_factor": 0.3}
    ucbvi43 = UCBVIAgent2(**ucbvi_params4)
    ucbvi43.fit(budget = fit_budget, seed = rand_seed)

    ucbvi_params44 = {"env": env4, "gamma": 1.0, "horizon": 10, "bonus_scale_factor": 0.1}
    ucbvi44 = UCBVIAgent2(**ucbvi_params4)
    ucbvi44.fit(budget = fit_budget, seed = rand_seed)

    ucbvi_params45 = {"env": env4, "gamma": 1.0, "horizon": 10, "bonus_scale_factor": 0.05}
    ucbvi45 = UCBVIAgent2(**ucbvi_params4)
    ucbvi45.fit(budget = fit_budget, seed = rand_seed)

    ucbvi_params46 = {"env": env4, "gamma": 1.0, "horizon": 10, "bonus_scale_factor": 0.01}
    ucbvi46 = UCBVIAgent2(**ucbvi_params4)
    ucbvi46.fit(budget = fit_budget, seed = rand_seed)

    regret1 = np.cumsum(optimal4.value - ucbvi4.value)
    #baseline_regret1 = np.cumsum(optimal1.value - baseline1.value)
    regret2 = np.cumsum(optimal4.value - ucbvi42.value)
    #baseline_regret2 = np.cumsum(optimal2.value - baseline2.value)
    regret3 = np.cumsum(optimal4.value - ucbvi43.value)
    #baseline_regret3 = np.cumsum(optimal3.value - baseline3.value)
    regret4 = np.cumsum(optimal4.value - ucbvi44.value)
    regret5 = np.cumsum(optimal4.value - ucbvi45.value)
    regret6 = np.cumsum(optimal4.value - ucbvi46.value)

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
    plt.savefig()

    np.save("./params/frozenlake_10_pi_star_1", regret1)
    np.save("./params/frozenlake_10_pi_star_1", regret1)
    np.save("./params/frozenlake_10_pi_star_1", regret1)
    np.save("./params/frozenlake_10_pi_star_1", regret1)
    np.save("./params/frozenlake_10_pi_star_1", regret1)
    np.save("./params/frozenlake_10_pi_star_1", regret1)


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