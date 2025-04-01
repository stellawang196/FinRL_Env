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

dow_ticker = YahooDownloader(
    start_date="2021-01-01", end_date="2023-12-01", ticker_list=["^DJI"]
).fetch_data()

returns = []
dow_ticker = dow_ticker["close"]
# Compute the returns
# for t in range(len(dow_ticker) - 1):
#     r_t = dow_ticker[t]
#     r_t_plus_1 = dow_ticker[t + 1]
#     return_t = (r_t_plus_1 - r_t) / r_t
#     returns.append(return_t)

# returns = np.array(returns)

# final_sharpe_ratio = sharpe_ratio(returns)
# final_max_drawdown = max_drawdown(returns)
# final_roma = return_over_max_drawdown(returns)

# print(
#     f"Max drawdown: {final_max_drawdown}, Final Sharpe: {final_sharpe_ratio}, Final ROMA: {final_roma}, Cum Return: {(dow_ticker[len(dow_ticker)-1] - dow_ticker[0]) / dow_ticker[0]}"
# )

initial_investment = 1000000  # Example initial investment
adj_close_prices = dow_ticker

# Calculate the daily returns
investment_values_list = []
initial_price = adj_close_prices.iloc[0]

for price in adj_close_prices:
    current_value = (price / initial_price) * initial_investment
    investment_values_list.append(current_value)


wandb.init(
    project="FinRL_emsemble_13_first_graph",
    config={},
    name=f"dow_run",
)
# wandb.log(
#     {
#         "cum reward": (dow_ticker[-1] - dow_ticker[0]) / dow_ticker[0],
#         "sharpe": final_sharpe_ratio,
#         "max_drawdown": final_max_drawdown,
#         "return_over_max_drawdown": final_roma,
#     }
# )
for rew in investment_values_list:
    wandb.log(
        {
            "ensemble_reward": rew,
        }
    )
