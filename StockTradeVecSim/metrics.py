import numpy as np
import pandas as pd
import empyrical as ep

def cumulative_returns(returns_pct):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return a pd.Series
    """
    return ep.cum_returns(returns_pct)

def sharpe_ratio(returns_pct, risk_free=0):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return float
    """
    returns = np.array(returns_pct)
    if returns.std() == 0:
        sharpe_ratio = np.inf
    else:
        sharpe_ratio = (returns.mean()-risk_free) / returns.std()
    return sharpe_ratio

def max_drawdown(returns_pct):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return: float
    """
    return ep.max_drawdown(returns_pct)

def return_over_max_drawdown(returns_pct):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return: float
    """
    mdd = abs(max_drawdown(returns_pct))
    returns = cumulative_returns(returns_pct)[len(returns_pct)-1]
    if mdd == 0:
        return np.inf
    return returns/mdd

def win_loss_rate(trades):
    """
    Calculates the win rate and loss rate from a series of trade outcomes.

    Parameters:
    - trades (list of floats): The profit or loss of each trade; positive for wins, negative for losses.

    Returns:
    - tuple: (win_rate, loss_rate) both expressed as percentages.
    """
    # Total number of trades
    total_trades = len(trades)
    
    # Counting wins and losses
    wins = sum(1 for result in trades if result > 0)
    losses = sum(1 for result in trades if result < 0)
    
    # Calculating win rate and loss rate
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    loss_rate = (losses / total_trades) * 100 if total_trades > 0 else 0
    
    return win_rate, loss_rate
