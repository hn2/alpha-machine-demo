from enum import Enum


class online_algorithm(Enum):
    PPO1 = 1
    PPO2 = 2
    SAC = 3
    TD3 = 4


class offline_algorithm(Enum):
    REM = 1
    BCQ = 2
    BEAR = 3
    AWR = 4
    ABM = 5
    CQL = 6
    AWAC = 7


class asset_class(Enum):
    equity = 1
    futures = 2
    options = 3
    forex = 4


class action_noise(Enum):
    none = 1
    normal = 2
    ornstein_uhlenbeck = 3


class compute_position(Enum):
    long_only = 1
    short_only = 2
    long_and_short = 3


class compute_indicators(Enum):
    prices = 1
    returns = 2
    log_returns = 3
    returns_hlc = 4
    log_returns_hlc = 5
    patterns = 6
    returns_patterns_volatility = 7
    momentum = 8
    all = 9
    misc = 10
    random = 11


class compute_reward(Enum):
    portfolio_value = 1
    returns = 2
    log_returns = 3

    # Risk-adjusted return based on Volatility
    treynor_ratio = 6
    sharpe_ratio = 7
    information_ratio = 8
    modigliani_ratio = 9

    # Risk-adjusted return based on Value at Risk
    excess_var = 10
    conditional_sharpe_ratio = 11

    # Risk-adjusted return based on Lower Partial Moments
    omega_ratio = 12
    sortino_ratio = 13
    kappa_three_ratio = 14
    gain_loss_ratio = 15
    upside_potential_ratio = 16

    # Risk-adjusted return based on Drawdown risk
    calmar_ratio = 17
    sterling_ratio = 18
    burke_ratio = 19


class reward_penalty(Enum):
    none = 1
    any = 2
    all = 3


class assets_allocation(Enum):
    equal_weighted = 1
    not_equal_weighted = 2


class compute_rebalance(Enum):
    method1 = 1
    method2 = 2
    method3 = 3


class account_currency(Enum):
    USD = 1
    EUR = 2


class lot_size(Enum):
    Nano = 100
    Micro = 1000
    Mini = 10000
    Standard = 100000
