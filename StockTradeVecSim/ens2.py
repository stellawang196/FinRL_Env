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

# Third-party imports
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb

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
from finrl.meta.preprocessor.preprocessors import data_split

# Local application/library specific imports
from StockTradeVecSim.env_vector_stocktrading import VectorizedStockTradingEnv


class VectorisedEnsembleEnv:
    # the ensemble strategy that we use is to compare the three policies against one another on a n day sliding
    # window basis and then use the one with the lowest sharpe ratio for each consecutive trading period

    def __init__(
        self,
        loaded_data,
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
        self.criteria = "sharpe"

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
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.tech_indicator_list = tech_indicator_list
        self.reward_scaling = reward_scaling
        self.num_envs = num_envs

        self.agent_stats = {policy: {} for policy in self.policies}
        self.agent_args = agent_args

        # create date ranges for train and test
        self.date_ranges = self._generate_date_ranges()

        self.risk_free_rates = {  # TODO use US bond interest rates
            "2020": 0.005,
            "2021": 0.015,
            "2022": 0.03,
            # "2023": 0.035,
        }

    def calculate_sharpe_ratio(self, rewards, window, rf_rate):
        # if len(rewards) != window: # TODO some mismatch with the dimensions of the first step
        #     raise ValueError("The length of rewards list must match the window length.")

        daily_rf_rate = (rf_rate / 252) * window
        p_return = (rewards[-1] - rewards[0]) / rewards[0]
        std_return = np.std(rewards)

        sharpe_ratio = (p_return - daily_rf_rate) / std_return
        return sharpe_ratio

    def ensemble(self, meta_learner=True):
        """
        Note: Because we reset the training env to the default starting conditions,
        we may lose some performance due to the continuous nature of the trading task
        """
        agents = []
        test_env = self._create_env(self.time_start, self.time_end, 1)
        validation_env = self._create_env(self.time_start, self.time_end, 1)
        ensemble_sharpe = []
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
            agents = [
                self._create_agents(
                    policy,
                    train_env.observation_space.shape[1],
                    train_env.action_space.shape[1],
                    self.agent_args,
                )
                for policy in self.policies
            ]

            for agent, policy in zip(agents, self.policies):
                print(f"Training {policy} agent...")
                trained_agent, train_rewards = self._train_agent(
                    train_env, agent, policy, train_start, train_end, self.max_episodes
                )

                validation_env.reset()
                validation_rewards = self._trade(
                    validation_env,
                    agent,
                    policy,
                    validation_start,
                    validation_end,
                )

                self.agent_stats[policy]["validation_sharpe_ratio"] = (
                    self.calculate_sharpe_ratio(
                        validation_rewards[0],
                        self.validation_window,
                        self.risk_free_rates[validation_start[0:4]],
                    )
                )

                self.agent_stats[policy]["validation_reward"] = validation_rewards[0][
                    -1
                ]
                self.agent_stats[policy]["validation_drawdown"] = max(
                    validation_rewards[0]
                ) - min(validation_rewards[0])

                # Save the trained agent
                save_path = f"./{policy}_agent"
                os.makedirs(save_path, exist_ok=True)
                trained_agent.save_or_load_agent(save_path, if_save=True)

                agents.append(trained_agent)

            # TRADING Use some method to decide which weighting to use
            if self.criteria == "sharpe":
                best_policy = max(
                    self.agent_stats,
                    key=lambda k: self.agent_stats[k]["validation_sharpe_ratio"],
                )
            elif self.criteria == "pmax":
                best_policy = max(
                    self.agent_stats,
                    key=lambda k: self.agent_stats[k]["validation_reward"],
                )
            elif self.criteria == "min_drawdowm":
                best_policy = max(
                    self.agent_stats,
                    key=lambda k: self.agent_stats[k]["validation_drawdown"],
                )

            print(f"Best agent for this period: {best_policy}")
            best_agent = self._create_agents(
                best_policy,
                test_env.observation_space.shape[1],
                test_env.action_space.shape[1],
                self.agent_args,
            )

            best_agent.save_or_load_agent(f"./{best_policy}_agent", if_save=False)

            test_rewards = self._trade(
                test_env, best_agent, best_policy, test_start, test_end
            )

            ensemble_sharpe.append(
                self.calculate_sharpe_ratio(
                    test_rewards[0][-self.test_window :],
                    self.test_window,
                    self.risk_free_rates[test_start[0:4]],
                )
            )

            print(
                f"{self.test_window} day rewards from trading period: {test_rewards[-self.test_window:]}"
            )

        rew_copy = test_rewards[0].tolist().copy()
        yearly_sharpe = []
        for year, rate in self.risk_free_rates.items():
            yearly_sharpe.append(
                self.calculate_sharpe_ratio(
                    rew_copy[:252],
                    252,
                    rate,
                )
            )
            del rew_copy[:252]

        # calculate daily and monthly sharpe ratio
        return test_rewards[0], yearly_sharpe

    def _train_agent(self, env, agent, policy, start_date, end_date, max_episodes):

        # Training settings
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        horizon_len = self.train_window

        # set agent settings
        agent.batch_size = horizon_len

        train_rewards = []

        # Training loop with tqdm and timing
        for _ in tqdm(range(max_episodes), desc="Training Progress", unit="episode"):
            # init env
            states = env.reset()
            agent.last_state = states[0].clone().detach()

            if policy == "PPO" or policy == "A2C":
                states, actions, logprobs, rewards, undones = agent.explore_vec_env(
                    env, horizon_len
                )
                agent.update_net((states, actions, logprobs, rewards, undones))
            elif policy == "DDPG" or policy == "SAC":
                states, actions, rewards, undones = agent.explore_vec_env(
                    env, horizon_len
                )
                buff = ReplayBuffer(
                    max_size=horizon_len,
                    state_dim=env.observation_space.shape[1],
                    action_dim=env.action_space.shape[1],
                    # num_envs=env.num_envs,
                )
                buff.update((states, actions, rewards, undones))
                agent.update_net(buff)

        # return episode_duration, samples, episode_final_assets, days
        train_rewards.extend(rewards.cpu().flatten())

        return agent, train_rewards

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
            print(policy)
            states, actions, logprobs, rewards, undones = agent.explore_vec_env(
                env, horizon_len
            )
            agent.update_net((states, actions, logprobs, rewards, undones))
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
            agent.update_net(buff)

        days, _, asset_mem = env.render(mode="log")

        return asset_mem  # , date_list

    def _create_env(self, start, end, num_envs):
        # takes in a date range to avoid making each env too big
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


if __name__ == "__main__":

    loaded_data = pd.read_csv("train_data.csv")

    env_args = {
        "hmax": 100,
        "initial_amount": 1_000_000,
        "tech_indicator_list": INDICATORS,
        "reward_scaling": 1e-4,
    }

    # env_counts = [2**x for x in range(0, 12, 4)]
    env_counts = [2**12]

    for num_envs in env_counts:
        wandb.init(
            project="FinRL_rewards_per_env",
            config={},
            name=f"{num_envs}",
        )
        wandb.define_metric("episodes")
        # define which metrics will be plotted against it
        wandb.define_metric("Portfolio_gain", step_metric="episodes")

        for max_ep in range(1, 51, 10):
            policies = ["PPO"]
            # policies = ["PPO", "DDPG", "SAC"]
            ensemble_args = {
                "loaded_data": loaded_data,
                # "policies": ["PPO", "DDPG", "SAC"],
                "policies": policies,
                "num_agents": len(policies),
                "max_episodes": max_ep,  # set to 1 for testing
                "time_start": "2020-01-01",
                "time_end": "2023-01-01",  # data ends here
                "train_window": 30,
                "validation_window": 5,
                "test_window": 5,
                "num_envs": num_envs,
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

            ensemble_rewards, ensemble_sharpe = (
                ensemble_env.ensemble()
            )  # 252 trading days in a year 3 yrs = 756
            # compute daily and monthly sharpe ratio
            # TODO compute the daily, monthly sharpe ratios

            # UNCOMMENT FOR ENSEMBLE TESTING

            wandb.log(
                {
                    "episodes": max_ep,
                    "Portfolio_gain": (ensemble_rewards[-1] - ensemble_rewards[0])
                    / ensemble_rewards[0],
                }
            )

        wandb.finish()
        # for rew in ensemble_rewards:
        #     wandb.log(
        #         {
        #             "Portfolio value": rew,
        #         }
        #     )
    # for sharpe in ensemble_sharpe:
    #     wandb.log(
    #         {
    #             "ensemble_sharpe": sharpe,
    #         }
    #     )

    # wandb.finish()

    # policies = ["PPO", "DDPG", "A2C"]
    # for p in policies:
    #     wandb.init(
    #         project="FinRL_emsemble_4",
    #         config={},
    #         name=f"Single_env_run_{p}",
    #     )
    #     ensemble_env.single_policy(p)
    #     wandb.finish()

    # TESTING FOR DIFFERENT ENV COUNTS


# btw, for ensemble strategy, I think providing a
# table to comapre cummulative returns, volatility,
# sharpe ratio, max drawdown for ensemble strategy
# and other single agents. This can show the
# efficiency of ensemble from different perspectives.
