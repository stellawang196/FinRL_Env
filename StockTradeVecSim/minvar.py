import wandb
import yfinance as yf
import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

from metrics import *


def fetch_latest_prices(data):
    """Fetch the latest available prices for each stock safely."""
    if not data.empty:
        latest_date = data.index.max()
        if latest_date in data.index:
            return data.loc[latest_date]
        else:
            print(f"Warning: Latest date {latest_date} not found in the index.")
            return None
    else:
        print("Warning: Data is empty.")
        return None


def fetch_data(tickers, start_date, end_date):
    """Fetch historical stock prices data from Yahoo Finance."""
    data = yf.download(tickers, start=start_date, end=end_date)
    return data["Adj Close"]


def calculate_returns(data):
    """Calculate daily log returns for each stock."""
    returns = np.log(data / data.shift(1))
    return returns.dropna()


def minimum_variance_portfolio(returns):
    """Calculate the weights for the minimum variance portfolio."""
    covariance_matrix = returns.cov().values
    n = len(covariance_matrix)
    w = cp.Variable(n, nonneg=True)
    portfolio_variance = cp.quad_form(w, covariance_matrix)
    constraints = [cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
    problem.solve()
    return w.value


def allocate_cash(weights, total_cash, latest_prices):
    """Allocate cash based on the portfolio weights and latest stock prices."""
    allocations = weights * total_cash
    number_of_shares = allocations / latest_prices
    return allocations, number_of_shares


def simulate_portfolio(data, shares):
    """Simulate the portfolio's value over time based on the number of shares owned."""

    portfolio_values = (data * shares).sum(axis=1)
    portfolio_values = portfolio_values.iloc[::-1]

    return portfolio_values.reset_index(drop=True)


def main():
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

    starting_cash = 1e6
    data_weights = YahooDownloader(
        start_date="2020-01-01", end_date="2021-01-01", ticker_list=dow_tickers
    ).fetch_data()
    data = YahooDownloader(
        start_date="2021-01-01", end_date="2024-01-01", ticker_list=dow_tickers
    ).fetch_data()

    pivoted_data_weights = data_weights.pivot(
        index="date", columns="tic", values="close"
    )
    pivoted_data = data.pivot(index="date", columns="tic", values="close")

    returns = calculate_returns(pivoted_data_weights)
    weights = minimum_variance_portfolio(returns)
    latest_prices = fetch_latest_prices(pivoted_data)
    allocations, number_of_shares = allocate_cash(weights, starting_cash, latest_prices)

    print("Portfolio Allocation:")
    for ticker, weight, allocation, shares in zip(
        dow_tickers, weights, allocations, number_of_shares
    ):
        print(
            f"{ticker}: Weight: {weight:.4f}, Allocation: ${allocation:.2f}, Shares: {shares:.4f}"
        )

    portfolio_values = simulate_portfolio(pivoted_data, number_of_shares)

    # print(portfolio_values)
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values, label="Portfolio Value", color="blue")
    plt.title("Portfolio Value Over Time")
    # plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    # plt.legend()
    # plt.grid(True)
    plt.savefig("test.png")
    plt.show()

    # compute metrics on min-var now

    returns = []

    ensemble_rewards = portfolio_values

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
        name=f"minvar_run",
    )
    wandb.log(
        {
            "cum reward": (
                ensemble_rewards[len(ensemble_rewards) - 1] - ensemble_rewards[0]
            )
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


if __name__ == "__main__":
    main()
