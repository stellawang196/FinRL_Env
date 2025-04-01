"""
This file executes training code in the vectorized environment and measures sampling rate

Authors: Nikolaus Holzer, Keyi Wang
Date: June 2024
"""

# Standard library imports
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
# import torch
# from tqdm import tqdm
import matplotlib.pyplot as plt

# Third-party imports
# from stable_baselines3 import PPO
# from stable_baselines3.common.logger import configure
# from stable_baselines3.common.vec_env import DummyVecEnv
# import wandb

# elegantrl library imports
# from elegantrl import train_agent, train_agent_multiprocessing
from elegantrl.agents import AgentPPO, AgentA2C, AgentDDPG
# from elegantrl.train.config import Config
from elegantrl import Config, get_gym_env_args

# finrl library imports
# from finrl.agents.stablebaselines3.models import DRLAgent
# from finrl.config import INDICATORS  # , TRAINED_MODEL_DIR, RESULTS_DIR
# from finrl.main import check_and_make_directories
# from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
# from finrl.meta.preprocessor.preprocessors import data_split

# Local application/library specific imports
from env_vector_stocktrading import VectorizedStockTradingEnv

# Constants
TRAIN_START_DATE = "2020-01-01"
TRAIN_END_DATE = "2022-01-01"
TEST_START_DATE = "2022-01-01"
TEST_END_DATE = "2023-01-01"
NUM_ENVS = 2 ** 11

INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]

def data_split(df, start, end, target_date_col="date"):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data

# Set horizon len to the number of days
start_date = datetime.strptime(TRAIN_START_DATE, "%Y-%m-%d")
end_date = datetime.strptime(TRAIN_END_DATE, "%Y-%m-%d")
HORIZON_LEN = (end_date - start_date).days


def make_env(num_envs: int = 1, start=TRAIN_START_DATE, end=TRAIN_END_DATE):
    # Load training data
    loaded_data = pd.read_csv(
        "/Users/nikh/Columbia/parallel_finrl/StockTradeVecSim/train_data.csv"
    )
    train_data = data_split(loaded_data, start, end)

    # Environment setup
    stock_dimension = len(train_data.tic.unique())
    state_space = (
            1
            + 1
            + stock_dimension
            + (5 * stock_dimension)
            + (len(INDICATORS) * stock_dimension)
    )
    buy_sell_cost = [0.001] * stock_dimension

    env_kwargs = {
        "df": train_data,
        "hmax": 100,
        "initial_amount": 1_000_000,
        "num_stock_shares": [0] * stock_dimension,
        "buy_cost_pct": buy_sell_cost,
        "sell_cost_pct": buy_sell_cost,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        "num_envs": num_envs,
    }

    # Vectorized environment for training
    env = VectorizedStockTradingEnv(**env_kwargs)

    return env


def make_agent(env, num_envs, policy="PPO"):
    # TODO make the dims get passed in as config arguments instead, probably best to abstract out logic from env making step
    # PPO Parameters
    """Possible agents: PPO, DDPG, A2C -> All actor critic models"""
    PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.0003,
        "batch_size": 64,
        "num_envs": num_envs,
    }

    config = Config()
    config.num_envs = num_envs

    # Network and agent configuration
    net_dims = [256, 256]  # MLP with 256 units in each of two hidden layers
    state_dim = env.observation_space.shape[1]
    action_dim = env.action_space.shape[1]

    if policy == "PPO":
        agent = AgentPPO(net_dims, state_dim, action_dim, gpu_id=0, args=config)
    elif policy == "DDPG":
        agent = AgentDDPG(net_dims, state_dim, action_dim, gpu_id=0, args=config)
    elif policy == "A2C":
        agent = AgentA2C(net_dims, state_dim, action_dim, gpu_id=0, args=config)

    return agent


def samples_per_second(env, agent):
    # Training

    steps = HORIZON_LEN
    horizon_len = 2
    episode_final_assets = 0
    episode_duration = 0
    samples = 0
    final_assets = []
    agent.batch_size = 1

    data = {
        "Step": [],
        "Reward_per_Second": [],
    }

    for step in range(steps):

        # init env
        if step == 0:
            states = env.reset()

        agent.last_state = states[0].clone().detach()

        start_time = time.time()
        states, actions, logprobs, rewards, undones = agent.explore_vec_env(
            env, horizon_len
        )
        agent.update_net((states, actions, logprobs, rewards, undones))
        episode_duration = time.time() - start_time

        # Fetch total assets
        _, _, asset_mem = env.render(mode="log")
        samples = rewards.shape[0] * rewards.shape[1]

        data["Step"].append(step)
        data["Reward_per_Second"].append(samples / episode_duration)

    return data


def plot_sample_rate(filepath, vecenvs):
    data = pd.read_pickle(filepath)
    colors = plt.cm.tab20(np.linspace(0, 1, len(vecenvs)))

    plt.figure(figsize=(8, 6))
    for i, vecenv in enumerate(vecenvs):
        env_data = data[f"{vecenv}"]

        plt.plot(
            env_data["Step"],
            env_data["Reward_per_Second"],
            label=f"{vecenv}",
            color=colors[i],
            # alpha=0.7,
            linewidth=1,
        )

    # Add titles and labels
    plt.title("Samples per second for various env counts", fontsize=18)
    plt.xlabel("Step", fontsize=18)
    plt.ylabel("Samples per Second", fontsize=18)
    plt.tick_params(axis="both", which="major", labelsize=14)

    legend = plt.legend(
        fontsize=14,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        title="Env count",
        frameon=False,
    )

    legend.get_title().set_fontsize("16")

    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("./output/stock_sample_rate_vecenvs.png")


if __name__ == "__main__":
    pkl_filename = "./output/env_performance_data.pkl"

    env_counts = [2 ** x for x in range(0, 12)]

    # if not os.path.exists(pkl_filename):
    #     data = {}
    #     for env_count in env_counts:
    #         env = make_env(
    #             num_envs=env_count, start=TRAIN_START_DATE, end=TRAIN_END_DATE
    #         )
    #         agent = make_agent(env, num_envs=env_count)
    #         env_data = samples_per_second(env, agent)

    #         data[f"{env_count}"] = env_data

    #     results_df = pd.DataFrame(data)
    #     results_df.to_pickle(pkl_filename)

    plot_sample_rate(pkl_filename, env_counts)
