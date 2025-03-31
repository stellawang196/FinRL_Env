from __future__ import annotations

from typing import List

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
from finrl.config import INDICATORS

import torch
from finrl.meta.preprocessor.preprocessors import data_split

from stable_baselines3.common.env_checker import check_env


matplotlib.use("Agg")

# from stable_baselines3.common.logger import Logger, KVWriter, CSVOutputFormat


class VectorizedStockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots: bool = False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
        num_envs: int = 1,
    ):
        self.num_envs = num_envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount  # get the initial cash
        self.buy_cost_pct = torch.tensor(
            buy_cost_pct, dtype=torch.float32, device=self.device
        )
        self.sell_cost_pct = torch.tensor(
            sell_cost_pct, dtype=torch.float32, device=self.device
        )
        self.reward_scaling = reward_scaling
        self.int_state_space = state_space
        self.int_action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.num_envs, action_space)
        )
        self.env_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_envs,))

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_envs, state_space)
        )
        self.data = self.df.loc[self.day, :]
        self.tickers = df["tic"].unique()
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration

        # env variables
        self.cash_balance = torch.full(
            (num_envs, 1), self.initial_amount, device=self.device
        )

        self.shares = torch.zeros(
            (num_envs, self.stock_dim), dtype=torch.float32, device=self.device
        )

        self.ohlcv = torch.tensor(
            np.array(
                [
                    self.df[self.df["tic"] == tic]
                    .iloc[0][["open", "high", "low", "close", "volume"]]
                    .values
                    for tic in self.tickers
                ],
                dtype=np.float32,
            ),
            device=self.device,
        )

        self.tech_inds = torch.tensor(
            np.array(
                [
                    self.df[self.df["tic"] == tic].iloc[0][INDICATORS].values
                    for tic in self.tickers
                ],
                dtype=np.float32,
            ),
            device=self.device,
        )

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # self.data = self.df.loc[self.day, :]

        self.rewards_memory = torch.zeros((self.num_envs, 1), device=self.device)
        self.actions_memory = torch.zeros((self.num_envs, 1), device=self.device)
        self.asset_memory = torch.zeros((self.num_envs, 1), device=self.device)

        self.average_reward = []
        self.average_assets = []

        # Initialize state
        self.state = self._initialize_state()

    def render(self, mode="human", close=False):
        if mode == "human":
            print("\n----- Episode Summary -----")
            print(f"Day: {self.day}")
            print(f"Number of envs: {self.num_envs}")

            # Calculate and display metrics for rewards
            rewards_summary = {
                "Max Reward": torch.max(self.rewards_memory).cpu().numpy(),
                "Min Reward": torch.min(self.rewards_memory).cpu().numpy(),
                "Median Reward": torch.median(self.rewards_memory).cpu().numpy(),
                "Mean Reward": torch.mean(self.rewards_memory).cpu().numpy(),
            }
            print("\n-- Rewards Metrics --")
            for key, value in rewards_summary.items():
                print(f"{key}: {value:.2f}")

            # Calculate and display metrics for total assets
            asset_summary = {
                "Max Total Assets": torch.max(self.asset_memory).cpu().numpy(),
                "Min Total Assets": torch.min(self.asset_memory).cpu().numpy(),
                "Median Total Assets": torch.median(self.asset_memory).cpu().numpy(),
                "Mean Total Assets": torch.mean(self.asset_memory).cpu().numpy(),
            }
            print("\n-- Total Assets Metrics --")
            for key, value in asset_summary.items():
                print(f"{key}: {value:.2f}")

            # Plot asset memory
            # plt.figure(figsize=(10, 4))
            # plt.title("Portfolio Value Over Time")
            # plt.plot(self.asset_memory.cpu().numpy().T)
            # plt.xlabel("Days")
            # plt.ylabel("Total Asset Value")
            # plt.grid(True)
            # plt.show()
        if mode == "log":
            return (
                self.day,
                self.rewards_memory.cpu().numpy(),
                self.asset_memory.cpu().numpy(),
            )
        if mode == "state":
            return self.state

    def step(self, actions):
        """Use vmap to apply the mapped buy and sell actions to all environments
        Change operations to use vectors more"""
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            return (
                self.state,
                self.reward,
                self.terminal,
                {},
            )

        self.day += 1
        self.update_prices()
        # TODO DDPG agent has different dimensions than others
        if actions.ndim == 1:
            print(f"Warning, action dim mismatch with agent. Got {actions.shape}...")
            actions = actions.unsqueeze(0)
        actions = actions * self.hmax
        actions = actions.to(dtype=torch.int32, device=self.device)

        if self.turbulence_threshold is not None:
            turbulence_mask = self.turbulence >= self.turbulence_threshold
            actions = torch.where(turbulence_mask, -self.hmax, actions)

        begin_total_asset = self._vmap_get_portfolio_value()

        buy_actions = torch.where(actions > 0, actions, torch.zeros_like(actions))
        sell_actions = torch.where(actions < 0, -actions, torch.zeros_like(actions))

        if buy_actions.any():
            self.cash_balance, self.shares = self._vmap_buy(
                self.cash_balance,
                self.shares,
                self.current_price,
                buy_actions,
                self.buy_cost_pct,
            )

        if sell_actions.any():
            self.cash_balance, self.shares = self._vmap_sell(
                self.cash_balance,
                self.shares,
                self.current_price,
                sell_actions,
                self.sell_cost_pct,
            )

        # After buy and sell, compute rewards
        end_total_assets = self._vmap_get_portfolio_value()
        reward = (end_total_assets - begin_total_asset) * self.reward_scaling

        # update state and env
        self.state = self._vmap_state()
        self.reward = reward.flatten()
        self.total_assets = end_total_assets

        # update logging
        self.asset_memory = torch.cat((self.asset_memory, end_total_assets), dim=1)
        self.rewards_memory = torch.cat((self.rewards_memory, reward), dim=1)

        return (
            self.state,
            self.reward,
            self.terminal,
            {},
        )

    def reset(self, seed=None, options=None):
        # reset state info to default
        self.day = 0
        self.cash_balance = torch.full(
            (self.num_envs, 1), self.initial_amount, device=self.device
        )
        self.shares = torch.zeros(
            (self.num_envs, self.stock_dim), dtype=torch.float32, device=self.device
        )

        # fetch new infos
        self.update_prices()
        self.state = self._initialize_state()

        # reset logging
        self.rewards_memory = torch.zeros((self.num_envs, 0), device=self.device)
        self.actions_memory = torch.zeros((self.num_envs, 0), device=self.device)
        self.asset_memory = torch.zeros((self.num_envs, 0), device=self.device)

        # return
        return self.state, {}

    def close(self):
        pass

    def update_prices(self):
        self.data = self.df.loc[self.day, :]
        self.current_price = (
            torch.tensor(
                self.data.close.values, dtype=torch.float32, device=self.device
            )
            .unsqueeze(0)
            .expand(self.num_envs, -1)
        )

    # env init functions
    def _initialize_state(self):
        """Creates a vectorized state matrix
        [initial cash, portfolio, ohlcv for all tickers, indicators for all tickers]"""
        self.update_prices()  # get most up to date prices

        self.total_assets = self._vmap_get_portfolio_value()
        self.asset_memory[:, 0] = self.total_assets.squeeze()
        return self._vmap_state()

    # Helper functions for vmap operations
    def _vmap_state(self):
        def get_state(total, amount, shares, ohlcv, tech_ind):
            return torch.cat(
                (total, amount, shares, ohlcv.flatten(), tech_ind.flatten())
            )

        return torch.vmap(func=get_state, in_dims=(0, 0, 0, None, None), out_dims=0)(
            self.total_assets,
            self.cash_balance,
            self.shares,
            self.ohlcv,
            self.tech_inds,
        )

    def _vmap_get_portfolio_value(self):
        """Compute the current amount for each environment using the portfolio and share values"""

        def calc_balance(cash_amount, shares, current_price):
            return (shares * current_price).sum() + cash_amount

        return torch.vmap(func=calc_balance, in_dims=(0, 0, None))(
            self.cash_balance, self.shares, self.current_price
        )

    def _vmap_buy(self, cash_balance, shares, prices, actions, buy_cost_pct):
        def buy(cash_balance, shares, prices, actions, buy_cost_pct):
            buy_costs = prices * (1 + buy_cost_pct)
            num_shares_to_buy = torch.min(cash_balance // buy_costs, actions)
            cash_balance -= (buy_costs * num_shares_to_buy).sum()
            shares += num_shares_to_buy.float()
            return cash_balance, shares

        new_cash_balance, new_shares = torch.vmap(func=buy, in_dims=(0, 0, 0, 0, None))(
            cash_balance.float(),
            shares.float(),
            prices.float(),
            actions.float(),
            buy_cost_pct,
        )
        return new_cash_balance, new_shares

    def _vmap_sell(self, cash_balance, shares, prices, actions, sell_cost_pct):
        def sell(cash_balance, shares, prices, actions, sell_cost_pct):
            sell_revenues = prices * (1 - sell_cost_pct)
            shares_to_sell = actions.abs() * shares
            shares_to_sell = torch.clamp(
                shares_to_sell, min=torch.zeros_like(shares), max=shares
            )
            cash_balance += (sell_revenues * shares_to_sell).sum()
            shares -= shares_to_sell
            return cash_balance, shares

        new_cash_balance, new_shares = torch.vmap(
            func=sell, in_dims=(0, 0, 0, 0, None)
        )(
            cash_balance.float(),
            shares.float(),
            prices.float(),
            actions.float(),
            sell_cost_pct,
        )
        return new_cash_balance, new_shares


def test_initiate_state(env):
    # Test reset
    state = env.reset()
    print(f"Initial State: with {env.num_envs} envs")
    print(state[0].shape)
    # print(state)


def sim_step(env):
    random_actions = (
        torch.rand((env.num_envs, env.stock_dim), device=env.device) * 2 - 1
    )

    # Print initial state for each environment
    # print("Initial State:")
    # print("Starting Cash Balances:", env.cash_balance[:, 0])
    # print("Initial Shares:", env.shares)

    # Execute one step in the environment
    state, reward, done, _ = env.step(random_actions)

    # Prepare the data for printing
    random_actions_cpu = random_actions * 100
    buy_actions = (random_actions > 0) * random_actions_cpu
    sell_actions = (random_actions < 0) * random_actions_cpu

    # Print actions for each environment
    # print("Actions taken (scaled):", random_actions_cpu)
    # # print("Shares bought or sold per stock:")
    # # print("Buy Actions (shares to buy):", buy_actions)
    # # print("Sell Actions (shares to sell):", sell_actions)

    # # Print final state for each environment
    # print("Final State after Step:")
    # print("Ending Cash Balances:", env.cash_balance[:, 0])
    # print("Ending total Balances:", env.total_assets[:, 0])
    # print("Final Shares:", env.shares)

    # print("New State Shape:", state.shape)
    # print("Reward:", reward)
    # print("Done Status:", done)


if __name__ == "__main__":
    # Create a dummy dataframe
    TRAIN_START_DATE = "2010-01-01"
    TRAIN_END_DATE = "2020-01-01"
    nenvs = 3

    processed_full = pd.read_csv("train_data.csv")
    train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)

    # Environment configs
    stock_dimension = len(train.tic.unique())
    # state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    state_space = (
        1
        + 1
        + stock_dimension
        + (5 * stock_dimension)
        + (len(INDICATORS) * stock_dimension)
    )

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        "num_envs": nenvs,
    }

    # TODO write a test function for state creation and reward computations

    env = VectorizedStockTradingEnv(df=train, **env_kwargs)
    test_initiate_state(env)
    env.render()
    # sim_step(env)
    # env.render()
    # sim_step(env)
    # env.render()
    # check_env(env)
