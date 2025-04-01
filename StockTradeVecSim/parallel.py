import torch
from stable_baselines3 import PPO
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3.common.logger import configure
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.main import check_and_make_directories
import wandb
import pandas as pd
from finrl.meta.preprocessor.preprocessors import data_split
from StockTradeVecSim.env_vector_stocktrading import VectorizedStockTradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv

# elegantrl stuff
from elegantrl.train.config import Config
from elegantrl.agents import AgentPPO


from tqdm import tqdm
import numpy as np

from elegantrl import train_agent, train_agent_multiprocessing
from elegantrl import Config, get_gym_env_args
from elegantrl.agents import AgentPPO, AgentDiscretePPO
from elegantrl.agents import AgentA2C, AgentDiscreteA2C

# from vector_PPO import AgentPPO

# from elegantrl.agents.AgentPPO import AgentPPO
# from elegantrl.run import train_and_evaluate_mp


# wandb.init(project="parallel-finrl-ppo")

TRAIN_START_DATE = "2010-01-01"
TRAIN_END_DATE = "2012-01-01"

# Load training data
loaded_data = pd.read_csv("train_data.csv")
train_data = data_split(loaded_data, TRAIN_START_DATE, TRAIN_END_DATE)

stock_dimension = len(train_data.tic.unique())
state_space = (
    1
    + 1
    + stock_dimension
    + (5 * stock_dimension)
    + (len(INDICATORS) * stock_dimension)
)
buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

num_envs = 2**11

env_kwargs = {
    "df": train_data,
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
    "num_envs": num_envs,
}


PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.0003,
    "batch_size": 1,
    "num_envs": num_envs,
}

config = Config()
config.num_envs = num_envs

# test using dummyvec env to vectorize as well

# TODO write code that runs the agent in the finrl environment versus in vectorized environment and compare metrics
# TODO use the test set to compute reward and total assets for final comparisons

env = VectorizedStockTradingEnv(**env_kwargs)
# env = VectorizedStockTradingEnv(**env_kwargs)
# env = DummyVecEnv([lambda: env])

net_dims = [256, 256]  # Example: MLP with 256 units in each of two hidden layers
state_dim = env.observation_space.shape[1]
action_dim = env.action_space.shape[1]

agent = AgentPPO(net_dims, state_dim, action_dim, gpu_id=0, args=config)

max_episodes = 10
horizon_len = 1000

episode_rewards = []
episode_total_assets = []

wandb.init(project="parallel finrl tests", config={})

for episode in tqdm(range(max_episodes), desc="Training Progress", unit="episode"):
    # TODO make each episode log results before calling reset
    states = env.reset()  # afaik explore vec env doesn't do reset
    agent.last_state = torch.tensor(states[0], dtype=torch.float32)

    states, actions, logprobs, rewards, undones = agent.explore_vec_env(
        env, horizon_len
    )
    agent.update_net((states, actions, logprobs, rewards, undones))

    # env.render(mode="human")

    # tqdm.write(f"Episode {episode+1}/{max_episodes} completed.")

days, rewards_memory, asset_memory = env.render(mode="log")

mean_rewards = np.max(rewards_memory, axis=0).tolist()
mean_portfolio_values = np.max(asset_memory, axis=0).tolist()

for i in range(days):
    wandb.log({"reward": mean_rewards[i], "assets": mean_portfolio_values[i]})
wandb.finish()
# torch.save(agent, "test.pth")
