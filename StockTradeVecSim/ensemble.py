"""
Ensemble learning for ElegantRL agents
Authors: Nik, Keyi
"""

# Standard library imports
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import argparse
import random
import string


# Third-party imports
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
import yfinance as yf

# elegantrl library imports
from elegantrl import train_agent, train_agent_multiprocessing
from elegantrl.agents import AgentPPO, AgentA2C, AgentDDPG, AgentSAC
from elegantrl.train.config import Config
from elegantrl import Config, get_gym_env_args
from elegantrl.train.replay_buffer import ReplayBuffer

# finrl library imports
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split

# Local application/library specific imports
from env_vector_stocktrading import VectorizedStockTradingEnv
from metrics import *
from kl_agents import AgentPPOKL


class VectorisedEnsembleEnv:
    # the ensemble strategy that we use is to compare the three policies against one another on a n day sliding
    # window basis and then use the one with the lowest sharpe ratio for each consecutive trading period

    def __init__(
        self,
        loaded_data,
        other_stocks,
        noisy_data,
        multi_data,
        policies: list[str],
        num_agents,
        max_episodes,
        time_start,
        time_end,
        train_window,
        validation_window,
        test_window,
        agent_args,
        hmax: int,
        initial_amount: int,
        tech_indicator_list: list[str],
        reward_scaling,
        num_envs,
    ):

        self.policies = policies
        self.multi_data = multi_data

        # training configs
        self.num_agents = num_agents
        self.max_episodes = max_episodes
        self.time_start = time_start
        self.time_end = time_end
        self.train_window = train_window
        self.validation_window = validation_window
        self.test_window = test_window

        # env configs
        self.loaded_data = loaded_data
        self.second_data = other_stocks
        self.noisy_data = noisy_data
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.tech_indicator_list = tech_indicator_list
        self.reward_scaling = reward_scaling
        self.num_envs = num_envs

        self.agent_stats = {policy: {} for policy in self.policies}
        self.agent_args = agent_args

        # additinoal noisy training data

        # create date ranges for train and test
        self.date_ranges = self._generate_date_ranges()

        self.risk_free_rates = {  # TODO use US bond interest rates
            "2020": 0.005,
            "2021": 0.015,
            "2022": 0.03,
            # "2023": 0.035,
        }

    def calculate_sharpe_ratio(self, returns_pct, risk_free=0.04):
        """
        returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

        return float
        """
        returns = np.array(returns_pct)
        if returns.std() == 0:
            sharpe_ratio = np.inf
        else:
            sharpe_ratio = (returns.mean() - risk_free) / returns.std()
        return sharpe_ratio

    def ensemble(self, simple=True):
        """
        Note: Because we reset the training env to the default starting conditions,
        we may lose some performance due to the continuous nature of the trading task
        """
        agents = []
        test_env = self._create_env(self.time_start, self.time_end, 1)
        validation_env = self._create_env(self.time_start, self.time_end, 1)
        ensemble_sharpe = []

        if len(self.policies) == 1:
            data_types = ["None"]

        if self.multi_data == True:
            data_types = (
                ["None"] * (len(self.policies) // 3)
                + ["second"] * (len(self.policies) // 3)
                + ["noise"] * (len(self.policies) // 3)
            )
        else:
            data_types = ["None"] * len(self.policies)

        for (
            train_start,
            train_end,
            validation_start,
            validation_end,
            test_start,
            test_end,
        ) in self.date_ranges:  # essentially step
            print(
                f"Training from {train_start} to {train_end}, Testing from {test_start} to {test_end}"
            )
            # TRAINING
            # env gets reset every time so no need for multiple
            train_env = self._create_env(train_start, train_end, self.num_envs)
            train_env_second = self._create_env(
                train_start, train_end, self.num_envs, "second"
            )
            train_env_noise = self._create_env(
                train_start, train_end, self.num_envs, "noise"
            )
            agents = [
                self._create_agents(
                    policy,
                    train_env.observation_space.shape[1],
                    train_env.action_space.shape[1],
                    self.agent_args,
                )
                for policy in self.policies
            ]
            trained_agents = []
            trained_sharpes = []

            """The code should be responsible for storing the previous action (pa) tensors and passing the correct one to the agent being trained"""
            previous_actions = []

            for agent, policy, data_type in zip(agents, self.policies, data_types):

                if self.multi_data:
                    if data_type == "second":
                        env_to_use = train_env_second
                    elif data_type == "noise":
                        env_to_use = train_env_noise
                else:
                    env_to_use = train_env

                print(f"Training {policy} agent...")

                # TODO train agent, save each action tensor
                trained_agent, train_rewards, train_actions = self._train_agent(
                    env_to_use,
                    agent,
                    policy,
                    train_start,
                    train_end,
                    self.max_episodes,
                    previous_actions,
                )

                previous_actions.extend(train_actions)

                if simple:
                    trained_sharpes = np.full(len(policies), 1 / len(policies))
                else:

                    validation_env.reset()
                    validation_rewards = self._trade(
                        validation_env,
                        agent,
                        policy,
                        validation_start,
                        validation_end,
                    )

                    returns = []

                    # Compute the returns
                    for t in range(len(validation_rewards[0].tolist()) - 1):
                        r_t = validation_rewards[0][t]
                        r_t_plus_1 = validation_rewards[0][t + 1]
                        return_t = (r_t_plus_1 - r_t) / r_t
                        returns.append(return_t)

                    agent_sharpe = self.calculate_sharpe_ratio(returns)
                    self.agent_stats[policy]["validation_sharpe_ratio"] = agent_sharpe

                    self.agent_stats[policy]["validation_reward"] = validation_rewards[
                        0
                    ][-1]
                    self.agent_stats[policy]["validation_drawdown"] = max(
                        validation_rewards[0]
                    ) - min(validation_rewards[0])

                    # Save the trained agent
                    save_path = f"./{policy}_agent"
                    os.makedirs(save_path, exist_ok=True)
                    trained_agent.save_or_load_agent(save_path, if_save=True)

                    trained_sharpes.append(agent_sharpe)

                trained_agents.append(trained_agent)

            temp = self.multi_trade(
                test_env,
                trained_agents,
                self.policies,
                trained_sharpes,
                test_start,
                test_end,
                simple,
            )

            print(temp)

        test_rewards = test_env.render(mode="log")

        return test_rewards[2][0]

    def _train_agent(
        self, env, agent, policy, start_date, end_date, max_episodes, previous_actions
    ):

        # Training settings
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        horizon_len = self.train_window

        # set agent settings
        agent.batch_size = horizon_len

        train_rewards = []
        train_actions = []

        # Training loop with tqdm and timing
        for _ in tqdm(range(max_episodes), desc="Training Progress", unit="episode"):
            # init env
            states = env.reset()
            if policy == "SAC":
                agent.last_state = states[0].clone().detach()
            else:
                agent.last_state = states[0].clone().detach()

            if policy == "PPO" or policy == "A2C":
                states, actions, logprobs, rewards, undones = agent.explore_vec_env(
                    env, horizon_len
                )
                train_actions.append(actions)
                if self.policies.count("PPO") > 1:
                    agent.update_net(
                        (states, actions, logprobs, rewards, undones), previous_actions
                    )
                else:
                    agent.update_net((states, actions, logprobs, rewards, undones))
            elif policy == "DDPG" or policy == "SAC":
                states, actions, rewards, undones = agent.explore_vec_env(
                    env, horizon_len
                )
                buff = ReplayBuffer(
                    max_size=horizon_len,
                    state_dim=env.observation_space.shape[1],
                    action_dim=env.action_space.shape[1],
                    # Some versions of replaybuffer use num envs, others use num_seqs. Need to check which one is being used
                    num_seqs=env.num_envs,
                    # num_envs=env.num_envs,
                )
                buff.update((states, actions, rewards, undones))
                train_actions.append(actions)
                agent.update_net(buff)

        # return episode_duration, samples, episode_final_assets, days
        train_rewards.extend(rewards.cpu().flatten())

        return agent, train_rewards, train_actions

    def multi_trade(
        self, env, agents, policies, sharpes, start_date, end_date, simple=True
    ):
        """In ensemble_train we train the meta learnier, all policies are used. Each agent explores the same vec env. The meta learner gets the action dim vector and returns a new single action dim vector. The meta learner then gets the reward from the env.

        We use teacher forcing, whilst the reward of the meta learner is less than the reward of the individual policy we replace the meta learner output with agent actions
        """

        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        horizon_len = (end_date - start_date).days
        date_list = pd.date_range(start=start_date, end=end_date).tolist()

        # TODO compute softmax array, then multiply inside of the loop each action by the correct softmax
        # TODO Only agent -1 which is SAC finally updates the env with the cumulative action vector

        exp_sharpes = np.exp(sharpes)
        sum_exp_sharpes = np.sum(exp_sharpes)
        softmax_weights = exp_sharpes / sum_exp_sharpes

        if simple:  # simple does a basic average calculation
            softmax_weights = np.full(softmax_weights.shape, 1 / len(policies))

        soft_actions = np.zeros((env.action_space.shape))

        for i, agent in enumerate(agents):
            policy = policies[i]
            # set agent settings
            agent.num_envs = 1
            agent.batch_size = horizon_len

            states = env.state
            if policy == "SAC":
                agent.last_state = states.clone().detach()
            else:
                agent.last_state = states[0].clone().detach()

            if policy == "PPO" or policy == "A2C":
                states, actions, logprobs, rewards, undones = agent.explore_vec_env(
                    env, horizon_len
                )

                soft_actions = np.add(soft_actions, actions * softmax_weights[i])

            elif policy == "DDPG" or policy == "SAC":
                states, actions, rewards, undones = agent.explore_vec_env(
                    env, horizon_len
                )

                soft_actions = np.add(soft_actions, actions * softmax_weights[i])

        # if policies[-1] == "PPO":
        #     agents[-1].update_net((states, soft_actions, logprobs, rewards, undones))
        # elif policies[-1] == "SAC":
        #     buff = ReplayBuffer(
        #         max_size=horizon_len,
        #         state_dim=env.observation_space.shape[1],
        #         action_dim=env.action_space.shape[1],
        #         # num_envs=env.num_envs,
        #     )

        #     buff.update((states, soft_actions, rewards, undones))
        #     agents[-1].update_net(buff)

        _, _, asset_mem = env.render(mode="log")

        return asset_mem  # , date_list

    def _trade(self, env, agent, policy, start_date, end_date):
        """In ensemble_train we train the meta learnier, all policies are used. Each agent explores the same vec env. The meta learner gets the action dim vector and returns a new single action dim vector. The meta learner then gets the reward from the env.

        We use teacher forcing, whilst the reward of the meta learner is less than the reward of the individual policy we replace the meta learner output with agent actions
        """

        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        horizon_len = (end_date - start_date).days
        date_list = pd.date_range(start=start_date, end=end_date).tolist()

        # set agent settings
        agent.num_envs = 1
        agent.batch_size = horizon_len

        states = env.state
        if policy == "SAC":
            agent.last_state = states.clone().detach()
        else:
            agent.last_state = states[0].clone().detach()

        if policy == "PPO" or policy == "A2C":
            states, actions, logprobs, rewards, undones = agent.explore_vec_env(
                env, horizon_len
            )
            # agent.update_net((states, actions, logprobs, rewards, undones), [])
        elif policy == "DDPG" or policy == "SAC":
            states, actions, rewards, undones = agent.explore_vec_env(env, horizon_len)

            # FOR SAC IT LOOKS LIKE THE CRI MODEL MIGHT BE MAKING SOME DIMENSION ISSUES

            buff = ReplayBuffer(
                max_size=horizon_len,
                state_dim=env.observation_space.shape[1],
                action_dim=env.action_space.shape[1],
                # num_envs=env.num_envs,
            )
            buff.update((states, actions, rewards, undones))
            # agent.update_net(buff)

        days, _, asset_mem = env.render(mode="log")

        return asset_mem  # , date_list

    def _create_env(self, start, end, num_envs, dataset=None):
        # takes in a date range to avoid making each env too big
        if dataset == "second":
            train_data = data_split(self.second_data, start, end)
        elif dataset == "noise":
            train_data = data_split(self.noisy_data, start, end)
        else:
            train_data = data_split(self.loaded_data, start, end)

        stock_dimension = len(train_data.tic.unique())
        state_space = (
            1
            + 1
            + stock_dimension
            + (5 * stock_dimension)
            + (len(INDICATORS) * stock_dimension)
        )
        buy_sell_cost = [0.001] * stock_dimension

        env_args = {
            "df": train_data,
            "hmax": self.hmax,
            "initial_amount": self.initial_amount,
            "num_stock_shares": [0] * stock_dimension,
            "buy_cost_pct": buy_sell_cost,
            "sell_cost_pct": buy_sell_cost,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": self.tech_indicator_list,
            "action_space": stock_dimension,
            "reward_scaling": self.reward_scaling,
            "num_envs": num_envs,
        }

        env = VectorizedStockTradingEnv(**env_args)
        return env

    def _create_agents(self, policy, state_dim, action_dim, agent_args):
        config = Config()
        config.num_envs = self.num_envs

        # Network and agent configuration
        net_dims = [256, 256]  # MLP with 256 units in each of two hidden layers
        # state_dim = self.train_envs[0].observation_space.shape[1]
        # action_dim = self.train_envs[0].action_space.shape[1]

        if policy == "PPO":
            if self.policies.count("PPO") > 1:
                agent = AgentPPOKL(
                    net_dims, state_dim, action_dim, gpu_id=0, args=config
                )
            else:
                agent = AgentPPO(net_dims, state_dim, action_dim, gpu_id=0, args=config)
        elif policy == "DDPG":
            agent = AgentDDPG(net_dims, state_dim, action_dim, gpu_id=0, args=config)
        elif policy == "A2C":
            agent = AgentA2C(net_dims, state_dim, action_dim, gpu_id=0, args=config)
        elif policy == "SAC":
            agent = AgentSAC(net_dims, state_dim, action_dim, gpu_id=0, args=config)

        return agent

    def _generate_date_ranges(self):
        """Generate an array of date ranges for training and testing."""
        date_ranges = []
        start_date = datetime.strptime(self.time_start, "%Y-%m-%d")
        end_date = datetime.strptime(self.time_end, "%Y-%m-%d")
        current_end = end_date

        while (
            current_end
            - pd.Timedelta(
                days=self.train_window + self.validation_window + self.test_window
            )
            > start_date
        ):
            test_end = current_end
            test_start = test_end - pd.Timedelta(days=self.test_window)
            validation_end = test_start - pd.Timedelta(days=1)
            validation_start = validation_end - pd.Timedelta(
                days=self.validation_window
            )
            train_end = validation_start - pd.Timedelta(days=1)
            train_start = train_end - pd.Timedelta(days=self.train_window)

            # Ensure we don't go before the start_date
            if train_start < start_date:
                train_start = start_date

            # Append the date ranges to the list
            date_ranges.append(
                (
                    train_start.strftime("%Y-%m-%d"),
                    train_end.strftime("%Y-%m-%d"),
                    validation_start.strftime("%Y-%m-%d"),
                    validation_end.strftime("%Y-%m-%d"),
                    test_start.strftime("%Y-%m-%d"),
                    test_end.strftime("%Y-%m-%d"),
                )
            )

            # Move current_end to the start of the next training window plus the test window minus one day to avoid overlapping
            current_end = test_start - pd.Timedelta(days=1)

        # Since we collected date ranges in reverse order, we need to reverse them before returning
        return date_ranges[::-1]

    def single_policy(self, policy="PPO"):
        """
        Executes a single policy over the specified date ranges.

        Args:
        - policy (str): The policy to be executed. Should be one of ['PPO', 'DDPG', 'A2C'].
        """
        date_ranges = self._generate_date_ranges()

        for train_start, train_end, _, _, test_start, test_end in date_ranges:
            print(
                f"Training from {train_start} to {train_end}, Testing from {test_start} to {test_end}"
            )

            # TRAINING
            train_env = self._create_env(train_start, train_end, self.num_envs)
            agent = self._create_agents(
                policy,
                train_env.observation_space.shape[1],
                train_env.action_space.shape[1],
                self.agent_args,
            )

            print(f"Training {policy} agent...")
            trained_agent, train_rewards = self._train_agent(
                train_env, agent, self.max_episodes
            )
            train_sharpe_ratio = self.calculate_sharpe_ratio(train_rewards)

            # Save the trained agent
            save_path = f"./{policy}_agent"
            os.makedirs(save_path, exist_ok=True)
            trained_agent.save_or_load_agent(save_path, if_save=True)

            # TRADING
            test_env = self._create_env(test_start, test_end, 1)
            test_rewards = self._trade(trained_agent, test_env, test_start, test_end)

        # Log the results, _trade returns asset_mem from render, so we can simply log the last
        for rew in test_rewards[0]:
            wandb.log(
                {
                    "ensemble_total_reward": rew,
                }
            )


def add_noise(tickers):
    """Reads in a ticker and distorts some of the data by adding a fixed number to all of the values"""


def generate_synthetic_ohlcv(existing_data):
    """
    Generate synthetic OHLCV data matching the dimensions of the provided dataset
    and scramble ticker symbols.

    Args:
    existing_data (pd.DataFrame): DataFrame containing the existing OHLCV data.

    Returns:
    pd.DataFrame: DataFrame containing synthetic OHLCV data with matching dimensions.
    """
    synthetic_ohlcv = existing_data.copy()

    # Generating synthetic OHLCV data
    n_rows = len(existing_data)
    synthetic_ohlcv["open"] = np.random.normal(loc=100, scale=10, size=n_rows)
    synthetic_ohlcv["close"] = synthetic_ohlcv["open"] + np.random.normal(
        loc=0, scale=2, size=n_rows
    )
    synthetic_ohlcv["high"] = np.maximum(
        synthetic_ohlcv["open"], synthetic_ohlcv["close"]
    ) + np.random.normal(loc=2, scale=1, size=n_rows)
    synthetic_ohlcv["low"] = np.minimum(
        synthetic_ohlcv["open"], synthetic_ohlcv["close"]
    ) - np.random.normal(loc=2, scale=1, size=n_rows)
    synthetic_ohlcv["volume"] = np.random.randint(1000, 10000, size=n_rows)

    # if "tic" in existing_data.columns:
    unique_tics = existing_data["tic"].unique()
    random_tics = {tic: generate_random_ticker() for tic in unique_tics}
    synthetic_ohlcv["tic"] = existing_data["tic"].map(random_tics)

    return synthetic_ohlcv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some policies.")
    parser.add_argument(
        "--policy", type=str, required=False, help="A single policy string."
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["PPO"],
        required=False,
        help="A list of policy strings.",
    )
    parser.add_argument("--noise", type=bool, required=False, default=False)
    args = parser.parse_args()

    # Access the arguments
    policy = args.policy
    policies = args.policies
    noise = args.noise
    less_popular_tickers = [
        "HBI",  # Hanesbrands Inc.
        "XRX",  # Xerox Holdings Corporation
        "FLS",  # Flowserve Corporation
        "UNM",  # Unum Group
        "NWL",  # Newell Brands Inc.
        "NOV",  # NOV Inc.
        "PNR",  # Pentair plc
        "RL",  # Ralph Lauren Corporation
        "NWSA",  # News Corp Class A
        "NWS",  # News Corp Class B
        "LEG",  # Leggett & Platt, Incorporated
        "BEN",  # Franklin Resources, Inc.
        "IPG",  # Interpublic Group of Companies, Inc.
        "SEE",  # Sealed Air Corporation
        "NI",  # NiSource Inc.
        "GPS",  # The Gap, Inc.
        "UAA",  # Under Armour, Inc. Class A
        "UA",  # Under Armour, Inc. Class C
        "PVH",  # PVH Corp.
        "IVZ",  # Invesco Ltd.
        "FRT",  # Federal Realty Investment Trust
        "VNO",  # Vornado Realty Trust
        "HOG",  # Harley-Davidson, Inc.
        "M",  # Macy's Inc
        "KIM",  # Kimco Realty Corporation
        "AIZ",  # Assurant, Inc.
        "ALK",
        "MHK",  # Mohawk Industries, Inc.
        "CPRI",  # Capri Holdings Limited
    ]

    # loaded_data = pd.read_csv("train_data.csv")
    # second_data = pd.read_csv("train_data.csv")
    # noisy_data = pd.read_csv("train_data.csv")
    # # second_data_raw = YahooDownloader(
    #     start_date="2019-11-01", end_date="2023-02-01", ticker_list=less_popular_tickers
    # ).fetch_data()

    dow_tickers = [
        "AAPL",
        "AMGN",
        "AXP",
        "BA",
        "CAT",
        "CRM",
        "CSCO",
        "CVX",
        "DIS",
        "DOW",
        "GS",
        "HD",
        "HON",
        "IBM",
        "INTC",
        "JNJ",
        "JPM",
        "KO",
        "MCD",
        "MMM",
        "MRK",
        "MSFT",
        "NKE",
        "PG",
        "TRV",
        "UNH",
        "V",
        "VZ",
        "WBA",
        "WMT",
    ]

    dow30_data_raw = YahooDownloader(
        start_date="2021-01-01", end_date="2024-01-01", ticker_list=dow_tickers
    ).fetch_data()

    dow_ticker = YahooDownloader(
        start_date="2021-01-01", end_date="2024-01-01", ticker_list=["DJI"]
    ).fetch_data()



    # noisy_data = generate_synthetic_ohlcv(second_data_raw)

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False,
    )

    dow30 = fe.preprocess_data(dow30_data_raw)

    # second_data = fe.preprocess_data(second_data_raw)
    # noisy_data = fe.preprocess_data(noisy_data)

    env_args = {
        "hmax": 100,
        "initial_amount": 1_000_000,
        "tech_indicator_list": INDICATORS,
        "reward_scaling": 1e-4,
    }

    # Some of the noisy data consits of training same basket for different time period.

    ensemble_args = {
        "loaded_data": dow30,
        "other_stocks": dow30,
        "noisy_data": dow30,
        "multi_data": False,
        "policies": policies,
        "num_agents": len(policies),
        "max_episodes": 1,  # set to 1 for testing
        "time_start": "2021-01-01",
        "time_end": "2023-12-01",  # set to 2021 for testing
        "train_window": 30,
        "validation_window": 5,
        "test_window": 5,
        "num_envs": 16,  # set to 2 for testing
    }

    agent_args = {
        "learning_rate": 0.0003,
        "batch_size": 64,
        "n_steps": 2048,
        "ent_coef": 0.01,
    }

    ensemble_env = VectorisedEnsembleEnv(
        **ensemble_args, **env_args, agent_args=agent_args
    )

    ensemble_rewards = ensemble_env.ensemble(simple=True)
    # 252 trading days in a year 3 yrs = 756

    returns = []

    # Compute the returns
    for t in range(len(ensemble_rewards) - 1):
        r_t = ensemble_rewards[t]
        r_t_plus_1 = ensemble_rewards[t + 1]
        return_t = (r_t_plus_1 - r_t) / r_t
        returns.append(return_t)

    returns = np.array(returns)

    final_sharpe_ratio = sharpe_ratio(returns)
    final_max_drawdown = max_drawdown(returns)
    final_roma = return_over_max_drawdown(returns)

    wandb.init(
        project="FinRL_emsemble_13_first_graph",
        config={},
        name=f"{policy}_run",
    )
    wandb.log(
        {
            "cum reward": (ensemble_rewards[-1] - ensemble_rewards[0])
            / ensemble_rewards[0],
            "sharpe": final_sharpe_ratio,
            "max_drawdown": final_max_drawdown,
            "return_over_max_drawdown": final_roma,
        }
    )
    for rew in ensemble_rewards:
        wandb.log(
            {
                "ensemble_reward": rew,
            }
        )

# btw, for ensemble strategy, I think providing a
# table to comapre cummulative returns, volatility,
# sharpe ratio, max drawdown for ensemble strategy
# and other single agents. This can show the
# efficiency of ensemble from different perspectives.

# TODO remove DDPG from ensemble
