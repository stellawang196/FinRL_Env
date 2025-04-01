"""
This file executes training code in the vectorized environment and measures sampling rate

Authors: Nikolaus Holzer, Keyi Wang
Date: June 2024
"""

# Standard library imports
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Third-party imports
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb

# elegantrl library imports
from elegantrl import train_agent, train_agent_multiprocessing
from elegantrl.agents import AgentPPO, AgentA2C, AgentDDPG
from elegantrl.train.config import Config
from elegantrl import Config, get_gym_env_args

# finrl library imports
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import (
    StockTradingEnv,
    StockPortfolioEnv,
)
from finrl.meta.preprocessor.preprocessors import data_split

# Local application/library specific imports
from StockTradeVecSim.env_vector_stocktrading import VectorizedStockTradingEnv
from markowitz_port import MarkowitzAgent

# Constants
TRAIN_START_DATE = "2020-01-01"
TRAIN_END_DATE = "2022-01-01"
TEST_START_DATE = "2022-01-01"
TEST_END_DATE = "2023-01-01"
NUM_ENVS = 2**11

# Set horizon len to the number of days
start_date = datetime.strptime(TRAIN_START_DATE, "%Y-%m-%d")
end_date = datetime.strptime(TRAIN_END_DATE, "%Y-%m-%d")
HORIZON_LEN = (end_date - start_date).days


def make_env(num_envs: int = 1, start=TRAIN_START_DATE, end=TRAIN_END_DATE):
    # Load training data
    loaded_data = pd.read_csv("train_data.csv")
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
        "batch_size": 1,
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


def measure_sampling_rate(env, agent, max_episodes: int = 1, env_counts: int = 1):

    # Training settings
    horizon_len = HORIZON_LEN
    episode_final_assets = 0
    episode_duration = 0
    samples = 0

    # Training loop with tqdm and timing
    for _ in tqdm(range(max_episodes), desc="Training Progress", unit="episode"):
        # TODO want to time how long one step takes
        # init env
        states = env.reset()
        agent.last_state = states[0].clone().detach()

        # Start measuring training time
        start_time = time.time()
        states, actions, logprobs, rewards, undones = agent.explore_vec_env(
            env, horizon_len
        )
        agent.update_net((states, actions, logprobs, rewards, undones))
        episode_duration = time.time() - start_time
        # Finish measuring training time

        # Calculate sampling rate
        samples = rewards.shape[0] * rewards.shape[1]

    # Fetch total assets
    days, _, asset_mem = env.render(mode="log")
    episode_final_assets = np.mean(asset_mem[-1])

    print(f"samples: {samples}, steps: {days}, samples/step: {samples / days}")

    # wandb.log(
    #     {
    #         "final_episode_rewards": episode_final_assets,
    #         "episode_duration": episode_duration,
    #         "samples_per_second": samples / episode_duration,
    #         "samples per step": samples / days,
    #     },
    #     step=env_count,
    # )

    # return
    return agent, episode_duration, samples, episode_final_assets, days


def measure_reward_time(env, agent, max_seconds: int = 110):
    """
    This function trains the agent for a max number of seconds
    Args:
    Returns:
    """
    # Training
    seconds = [s for s in range(10, max_seconds, 10)]
    horizon_len = HORIZON_LEN
    episode_final_assets = 0
    episode_duration = 0
    samples = 0
    final_assets = []

    for secs in seconds:
        end_time = time.time() + secs
        print(f"Training for {secs} seconds...")
        while time.time() < end_time:
            # init env
            states = env.reset()
            agent.last_state = states[0].clone().detach()

            states, actions, logprobs, rewards, undones = agent.explore_vec_env(
                env, horizon_len
            )
            agent.update_net((states, actions, logprobs, rewards, undones))

        # Fetch total assets
        _, _, asset_mem = env.render(mode="log")
        episode_final_assets = np.mean(asset_mem[-1])

        wandb.log(
            {"reward_per_second": episode_final_assets},
        )

    return episode_final_assets


def samples_per_second(env, agent):
    # Training

    steps = HORIZON_LEN
    horizon_len = 2
    episode_final_assets = 0
    episode_duration = 0
    samples = 0
    final_assets = []
    agent.batch_size = 1

    data = {"Step": [], "Rewards_per_Second": [], "Final_Assets": []}

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
        episode_final_assets = np.mean(asset_mem[-1])
        samples = rewards.shape[0] * rewards.shape[1]

        wandb.log(
            {"reward_per_second": samples / episode_duration / 2},
            # divide by 2 since we are doing horizon len of 2 so double samples
        )

    return episode_final_assets


def test_agent(agent, env, horizon_len=HORIZON_LEN):
    """test the agent in a standard finrl stocktradingenv
    We run the agent for one episode to simulate a real world trading scenario"""

    print(f"Testing agent trained in a vec env with {agent.num_envs} envs...")

    agent.num_envs = 1

    states = env.reset()
    agent.last_state = states[0].clone().detach()

    states, actions, logprobs, rewards, undones = agent.explore_vec_env(
        env, horizon_len
    )
    agent.update_net((states, actions, logprobs, rewards, undones))

    days, _, asset_mem = env.render(mode="log")
    episode_final_assets = np.mean(asset_mem[-1])

    # wandb.log({"testing rewards": episode_final_assets})
    return episode_final_assets


def train_agent(env, agent, max_episodes):
    """
    Perform ensemble learning using PPO, DDPG, and A2C agents.
    Args:
    - agents: list of trained agent objects [ppo_agent, ddpg_agent, a2c_agent]
    - envs: list of vectorized environments [ppo_env, ddpg_env, a2c_env]
    - horizon_len: int, the number of steps for each episode
    """

    # Training settings
    horizon_len = HORIZON_LEN

    # Training loop with tqdm and timing
    for _ in tqdm(range(max_episodes), desc="Training Progress", unit="episode"):
        # init env
        states = env.reset()
        agent.last_state = states[0].clone().detach()

        states, actions, logprobs, rewards, undones = agent.explore_vec_env(
            env, horizon_len
        )
        agent.update_net((states, actions, logprobs, rewards, undones))

    # return episode_duration, samples, episode_final_assets, days
    return agent


if __name__ == "__main__":
    # TODO add args to select which experiments to run

    env_counts = [2**x for x in range(0, 12)]

    testing_env = make_env(num_envs=1, start=TEST_START_DATE, end=TEST_END_DATE)

    # wandb.define_metric("test/num_envs")
    # wandb.define_metric("test/*", step_metric="test/num_envs")

    for env_count in env_counts:
        wandb.init(
            project="FinRL_tests_4",
            config={},
            name=f"PPO_vecenv_{env_count}",
        )
        env = make_env(num_envs=env_count, start=TRAIN_START_DATE, end=TRAIN_END_DATE)
        agent = make_agent(env, num_envs=env_count)

        # measure sampling rate
        # we create a new agent for each env count. We can therefore keep the latest agent returned by one of the measuring funcs
        # testing_agent, episode_duration, samples, episode_final_assets, days = (
        #     measure_sampling_rate(env, agent, max_episodes=1)
        # )
        # time_total_assets = measure_reward_time(env, agent, max_seconds=110)

        # # now we test that agent on a three month time span
        # test_final_assets = test_agent(testing_agent, testing_env)
        # # wandb.log(
        # #     {
        # #         # "test/num_envs": env_count,  # x axis
        # #         "test/episode_duration": episode_duration,
        # #         "test/samples_per_second": samples / episode_duration,
        # #         "test/samples_per_step": samples / days,
        # #         "test/final_assets": test_final_assets,
        # #     },
        # # )
        samples_per_second(env, agent)
        wandb.finish()

        # FOR VARYING ENV COUNTS
        # TODO log samples per second over steps
        # TODO log portfolio value over steps

"""
Training settings
- num_envs = [2**x for x in range(12)]
- max_training_time = seconds = [s for s in range(10, 110, 10)]

The first experiment measures the sampling rate for different env counts. In a loop, create envs with num_envs (from the array). Samples per step and samples per second are measured. We can calculate theoretical samples per step by doing (num_envs * stock_dim). We measuer the samples per second by training the agent for max_training_time and counting the samples and dividing by the total seconds. We do this for various num_envs and log the samples per step and second.

We also test the reward on a test set for various env counts. Simply train an agent for each env count and make it trade in a testing env with num_envs = 1 for a set timeperiod. The goal is to compare how agents trained in different number of envs perform against each other.
"""
