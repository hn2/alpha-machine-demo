import numpy as np

from .pf_stats import pf_stats


# ['avg_loss',
#  'avg_return',
#  'avg_win',
#  'best',
#  'cagr',
#  'calmar',
#  'common_sense_ratio',
#  'comp',
#  'compare',
#  'compsum',
#  'conditional_value_at_risk',
#  'consecutive_losses',
#  'consecutive_wins',
#  'cpc_index',
#  'cvar',
#  'drawdown_details',
#  'expected_return',
#  'expected_shortfall',
#  'exposure',
#  'gain_to_pain_ratio',
#  'geometric_mean',
#  'ghpr',
#  'greeks',
#  'implied_volatility',
#  'information_ratio',
#  'kelly_criterion',
#  'kurtosis',
#  'max_drawdown',
#  'monthly_returns',
#  'outlier_loss_ratio',
#  'outlier_win_ratio',
#  'outliers',
#  'payoff_ratio',
#  'profit_factor',
#  'profit_ratio',
#  'r2',
#  'r_squared',
#  'rar',
#  'recovery_factor',
#  'remove_outliers',
#  'risk_of_ruin',
#  'risk_return_ratio',
#  'rolling_greeks',
#  'ror',
#  'sharpe',
#  'skew',
#  'sortino',
#  'adjusted_sortino',
#  'tail_ratio',
#  'to_drawdown_series',
#  'ulcer_index',
#  'ulcer_performance_index',
#  'upi',
#  'utils',
#  'value_at_risk',
#  'var',
#  'volatility',
#  'win_loss_ratio',
#  'win_rate',
#  'worst']

class reward_scheme_quantstats:

    def __init__(self,
                 number_of_instruments,
                 compute_reward):

        self.number_of_instruments = number_of_instruments
        self.compute_reward = compute_reward
        self.pf_stats = pf_stats()

    def get_reward(self, portfolio_returns, portfolio_log_returns):

        r = np.array(portfolio_returns)
        e = np.mean(r)
        m = np.zeros(self.number_of_instruments + 1)
        f = 0.0

        reward = 0

        for compute_reward in self.compute_reward:

            # return/log return
            if compute_reward == 'returns':
                reward += portfolio_returns[-1]
            elif compute_reward == 'log_returns':
                reward += portfolio_log_returns[-1]
            elif compute_reward == 'log_returns_max_dd':
                reward += (portfolio_log_returns[-1] - self.pf_stats.max_dd(r) / 100)

            # Risk-adjusted return based on Volatility
            elif compute_reward == 'treynor_ratio':
                reward += self.pf_stats.treynor_ratio(e, r, m, f)
            elif compute_reward == 'sharpe_ratio':
                reward += self.pf_stats.sharpe_ratio(e, r, f)
            elif compute_reward == 'information_ratio':
                reward += self.pf_stats.information_ratio(r, m)
            elif compute_reward == 'modigliani_ratio':
                reward += self.pf_stats.modigliani_ratio(e, r, m, f)

            # Risk-adjusted return based on Value at Risk
            elif compute_reward == 'excess_var':
                reward += self.pf_stats.excess_var(e, r, f, 0.05)
            elif compute_reward == 'conditional_sharpe_ratio':
                reward += self.pf_stats.conditional_sharpe_ratio(e, r, f, 0.05)

            # Risk-adjusted return based on Lower Partial Moments
            elif compute_reward == 'omega_ratio':
                reward += self.pf_stats.omega_ratio(e, r, f)
            elif compute_reward == 'sortino_ratio':
                reward += self.pf_stats.sortino_ratio(e, r, f)
            elif compute_reward == 'kappa_three_ratio':
                reward += self.pf_stats.kappa_three_ratio(e, r, f)
            elif compute_reward == 'gain_loss_ratio':
                reward += self.pf_stats.gain_loss_ratio(r)
            elif compute_reward == 'upside_potential_ratio':
                reward += self.pf_stats.upside_potential_ratio(r)

            # Risk-adjusted return based on Drawdown risk
            elif compute_reward == 'calmar_ratio':
                reward += self.pf_stats.calmar_ratio(e, r, f)
            elif compute_reward == 'sterling_ratio':
                reward += self.pf_stats.sterling_ratio(e, r, f, 5)
            elif compute_reward == 'burke_ratio':
                reward += self.pf_stats.burke_ratio(e, r, f, 5)

        #   print(f"reward={reward}")

        return reward
