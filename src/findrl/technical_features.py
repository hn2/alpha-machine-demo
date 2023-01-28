import random

import numpy as np
import talib

from .math_utils import math_utils


# from .pf_stats import pf_stats
# from .stats import sharpe, smart_sharpe, rolling_sharpe, sortino, smart_sortino, rolling_sortino, adjusted_sortino, \
#     omega, gain_to_pain_ratio, cagr, rar, kurtosis, calmar, ulcer_index, ulcer_performance_index, serenity_index, \
#     risk_of_ruin, value_at_risk, conditional_value_at_risk, tail_ratio, payoff_ratio, profit_ratio, profit_factor, \
#     cpc_index, common_sense_ratio, outlier_win_ratio, outlier_loss_ratio, recovery_factor, risk_return_ratio, \
#     max_drawdown, kelly_criterion, information_ratio


class technical_features:

    def __init__(self,
                 security,
                 open_name,
                 high_name,
                 low_name,
                 close_name):
        self.security = security
        self.open_name = open_name
        self.high_name = high_name
        self.low_name = low_name
        self.close_name = close_name

        self.open_price = self.security[self.open_name].values
        self.high_price = self.security[self.high_name].values
        self.low_price = self.security[self.low_name].values
        self.close_price = self.security[self.close_name].values

    def get_indicators_prices(self):
        #   Prices
        self.security['OPEN'] = self.open_price
        self.security['HIGH'] = self.high_price
        self.security['LOW'] = self.low_price
        self.security['CLOSE'] = self.close_price

        return self.security.dropna().astype(np.float)

    def get_indicators_returns(self):
        self.security['RR'] = self.security[self.close_name].fillna(1) / self.security[self.open_name].fillna(1)

        return self.security.dropna().astype(np.float)

    def get_indicators_log_returns(self):
        self.security['LOG_RR_1'] = np.log(self.security[self.close_name].fillna(1)) - np.log(
            self.security[self.open_name].fillna(1))

        return self.security.dropna().astype(np.float)