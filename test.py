import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from datetime import datetime
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
from envs.inventory import Inventory
from envs.frozenlake import FrozenLake
from utls.regret_writter import RandomAgent2, UCBVIAgent2, OptimalAgent
from rlberry.manager import AgentManager, plot_writer_data

if __name__ == '__main__':
    # Set environment
    env_ctor = FrozenLake # change for other environments
    i = 15
    k = 70
    t = 10
    prob = 0.2
    a = 2
    a2 = 6
    a3 = 9
    a4 = 12
    env_kwargs = dict(slip_prob=prob, I = i, K = k, T = t, alpha = a)
    env_kwargs2 = dict(slip_prob=prob, I = i, K = k, T = t, alpha = a2)
    env_kwargs3 = dict(slip_prob=prob, I = i, K = k, T = t, alpha = a3)
    env_kwargs4 = dict(slip_prob=prob, I = i, K = k, T = t, alpha = a4)
    env = env_ctor(**env_kwargs)

    # Define parameters
    ucbvi_params = {"gamma": 1.0, "horizon": 10}
    # Create AgentManager to fit 3 agents using 1 job
    ucbvi_stats = AgentManager(
        UCBVIAgent2,
        (env_ctor, env_kwargs),
        fit_budget=5,
        init_kwargs=ucbvi_params,
        n_fit=1,
        parallelization="process",
        seed=42,
        mp_context="fork",
    )  # mp_context is needed to have parallel computing in notebooks.
    ucbvi_stats.fit()

    '''
    # Create AgentManager to fit 3 agents using 1 job
    ucbvi_stats2 = AgentManager(
        UCBVIAgent2,
        (env_ctor, env_kwargs2),
        fit_budget=10,
        init_kwargs=ucbvi_params,
        n_fit=2,
        parallelization="process",
        seed=42,
        mp_context="fork",
    )  # mp_context is needed to have parallel computing in notebooks.
    #ucbvi_stats2.fit()
    '''
    
    '''
    # Create AgentManager for baseline
    baseline_stats = AgentManager(
    RandomAgent2,
    (env_ctor, env_kwargs),
    fit_budget=1000,
    n_fit=5,
    parallelization="process",
    mp_context="fork",
    seed=42,
)
    baseline_stats.fit()
    '''
    # Create AgentManager for baseline
    opti_stats = AgentManager(
        OptimalAgent,
        (env_ctor, env_kwargs),
        fit_budget=5,
        n_fit=1,
        parallelization="process",
        mp_context="fork",
        seed=42,
    )
    opti_stats.fit()

    '''
    # Create AgentManager for baseline
    opti_stats2 = AgentManager(
        OptimalAgent,
        (env_ctor, env_kwargs2),
        fit_budget=10,
        n_fit=2,
        parallelization="process",
        mp_context="fork",
        seed=42,
    )
    #opti_stats2.fit()
    '''
    df = plot_writer_data(opti_stats, tag="episode_rewards", show=False)
    df = df.loc[df["tag"] == "episode_rewards"][["global_step", "value"]]
    opti_reward1 = df.groupby("global_step").mean()["value"].values

    '''
    df = plot_writer_data(opti_stats2, tag="reward", show=False)
    df = df.loc[df["tag"] == "reward"][["global_step", "value"]]
    opti_reward2 = df.groupby("global_step").mean()["value"].values
    '''

    def compute_regret(rewards):
        return np.cumsum(opti_reward1 - rewards[: len(opti_reward1)])

    #def compute_regret2(rewards):
        #return np.cumsum(opti_reward2 - rewards[: len(opti_reward2)])

    # Plot of the cumulative reward.
    output = plot_writer_data(
        #[ucbvi_stats, baseline_stats, opti_stats],
        [ucbvi_stats, opti_stats],
        tag="episode_rewards",
        #xtag = "episode",
        preprocess_func= compute_regret, 
        title="Cumulative Regret",
    )

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