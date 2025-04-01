from collections import defaultdict
import cvxpy as cp

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
import config
import config_tickers
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from portfolio_env import StockPortfolioEnv

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


def data_split(df, start, end, target_date_col="date"):
    """
    split the dataset into training or testing using time
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    dates = pd.to_datetime(df[target_date_col])
    data = df[(dates >= start) & (dates < end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data


# example markowitz agent using info dict from environment to do optimization at each time step


class MarkowitzAgent:
    """Provides implementations for Markowitz agent (mean-variance optimization)
    Attributes
    ----------
        env: gym environment class
            user-defined class
    Methods
    -------
        prediction()
            make a prediction in a test dataset and get result history
    """

    def __init__(
        self,
        env,
        risk_aversion=10,
        annual_risk_free_rate=0.03,  # disregard risk free rate since RL disregards
    ):
        super().__init__()
        self.risk_aversion = risk_aversion
        self.env = env
        # compute daily risk free rate from annual risk free rate
        # self.risk_free_rate = (1 + annual_risk_free_rate) ** (1 / 365) - 1
        # disable risk free rate for now
        self.risk_free_rate = -1

    def get_model(self, model_name, model_kwargs):
        raise NotImplementedError()

    def train_model(self, model, cwd, total_timesteps=5000):
        raise NotImplementedError()

    def act(self, state, info):
        """
        This is the core of markowitz portfolio optimization
        it maximizes the éxpected_return - risk_aversion * risk
        with expected_return = mean_returns @ portfolio_weights
        and risk = portfolio_weights.T @ cov @ portfolio_weights
        The constraints say that the weights must be positive and sum to 1

        returns the action as the weights of the portfolio
        """
        # unpack state to get covariance and means
        data = info["data"].copy()
        # from the data estimate returns and covariances
        cov = data.iloc[-1].cov_list
        mean_returns = data[data["date"] == data["date"].max()][
            "ewm_returns"
        ].to_numpy()

        # solve markowitz model with cvxpy
        # initialize model
        num_stocks = len(mean_returns)
        weights = cp.Variable(num_stocks)
        risk_free_weight = cp.Variable(1)
        # define constraints
        # constraints = [cp.sum(weights) + risk_free_weight ==
        #                1, weights >= 0, risk_free_weight >= 0]
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0,
            #    risk_free_weight >= 0
        ]
        # define objective
        # + risk_free_weight*self.risk_free_rate
        portfolio_return = mean_returns @ weights
        portfolio_risk = cp.quad_form(weights, cov)
        # define objective
        objective = cp.Maximize(portfolio_return - self.risk_aversion * portfolio_risk)
        # define problem
        problem = cp.Problem(objective, constraints)
        # solve problem
        problem.solve()
        # get weights
        weights = weights.value
        # get action. if using risk free rate then integrate it into the action
        action = weights
        # action = np.concatenate([weights, risk_free_weight.value])
        action = np.maximum(action, 0)
        action = action / np.sum(action)
        return action

    def prediction(self, environment):
        # args = Arguments(env=environment)
        # args.if_off_policy
        # args.env = environment

        # test on the testing env
        state, info = environment.reset()
        day = environment.sorted_times[environment.time_index]
        history = defaultdict(list)

        total_asset = environment.portfolio_value
        history["date"].append(day)
        history["total_assets"].append(total_asset)
        history["episode_return"].append(0)
        # episode_total_assets.append(environment.initial_amount)
        done = False
        while not done:
            action = self.act(state, info)
            state, reward, done, trunc, info = environment.step(action)
            day = environment.sorted_times[environment.time_index]

            total_asset = environment.portfolio_value
            episode_return = total_asset / environment.initial_amount
            history["date"].append(day)
            history["total_assets"].append(total_asset)
            history["episode_return"].append(episode_return)
        print("Test Finished!")
        # return episode total_assets on testing data
        print("episode_return", episode_return)
        return pd.DataFrame(history)


def main(
    start_date=config.TRAIN_START_DATE,
    end_date=config.TRADE_END_DATE,
    ticker_list=config_tickers.DOW_30_TICKER,
    time_interval="1D",
    data_source="yahoofinance",
    technical_indicator_list=config.INDICATORS,
    if_vix=True,
    hmax=100,
    initial_amount=1000000,
    transaction_cost_pct=0.001,
    reward_scaling=1e-4,
):
    df = YahooDownloader(  # 1 year lookback
        start_date="2020-01-01", end_date="2024-01-01", ticker_list=dow_tickers
    ).fetch_data()

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False,
    )

    df = fe.preprocess_data(df)

    # Group by 'tic' without setting it as index
    df["pct_change"] = df.groupby("tic")["close"].pct_change()

    cov_list = []
    # look back is one year
    lookback = 252
    for i in range(lookback, len(df.index.unique())):
        data_lookback = df.loc[i - lookback : i, :]
        price_lookback = data_lookback.pivot_table(
            index="date", columns="tic", values="close"
        )
        return_lookback = price_lookback.pct_change(fill_method=None).dropna()
        covs = return_lookback.cov().values
        cov_list.append(covs)

    dates = df.date.unique()[lookback:]
    cov_list = cov_list[-len(dates) :]

    df["mean_pct_change_lookback"] = df.rolling(lookback)["pct_change"].mean()
    df["ewm_returns"] = df["pct_change"].ewm(span=50).mean()
    df_cov = pd.DataFrame({"date": df.date.unique()[lookback:], "cov_list": cov_list})
    df = df.merge(df_cov, on="date")
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)

    # trade_df = df
    test_df = data_split(df, start="2021-01-01", end="2024-01-01")

    stock_dimension = len(test_df.tic.unique())
    state_space = stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    env_kwargs = {
        "hmax": hmax,
        "initial_amount": initial_amount,
        "transaction_cost_pct": transaction_cost_pct,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": reward_scaling,
    }
    e_test_gym = StockPortfolioEnv(df=test_df, **env_kwargs)
    agent = MarkowitzAgent(e_test_gym)
    df_daily_return = agent.prediction(e_test_gym)

    # print(df_daily_return)

    wandb.init(
        project="FinRL_emsemble_13_first_graph",
        config={},
        name=f"markowitz_run",
    )
    for ret in df_daily_return["total_assets"]:
        wandb.log({"ensemble_reward": ret})


if __name__ == "__main__":
    """
    run this script by:
    python FinRL-Meta/examples/run_markowitz_portfolio_optimization.py
    """
    main()
