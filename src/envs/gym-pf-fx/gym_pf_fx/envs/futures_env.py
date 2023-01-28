#   https://github.com/yuriak/RLQuant/blob/master/env/stock_env.py

import random
from pprint import pprint

import gym
import gym.spaces as spaces
import numpy as np
import pandas as pd
from gym.utils import seeding

from .convert import convert
from .math_utils import math_utils
from .pf_stats import pf_stats
from .reward_scheme import reward_scheme
from .technical_features import technical_features

CURRENT_PORTFOLIO_VALUE = 0.8
MAX_DRAWDOWN = 0.2
SHARPE_RATIO = 1.5


class FxEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 data,
                 instruments,
                 lookback=10,
                 random_episode_start=True,
                 cash=1e6,
                 max_slippage_percent=1e-2,
                 lot_size='Standard',
                 leverage=2e1,
                 pip_size=[0.0001],
                 pip_spread=[3],
                 compute_position='long_and_short',  # long_only, short_only, long_and_short
                 compute_indicators='log_returns',
                 # prices, returns, log_returns, returns_hlc, log_returns_hlc, patterns, returns_patterns_volatility, momentum, all, misc, random
                 compute_reward='log_returns',  # returns log_returns
                 verbose=False):  #

        super(FxEnv, self).__init__()

        self.data = data
        self.instruments = instruments
        self.number_of_instruments = len(self.instruments)
        self.lookback = lookback
        self.random_episode_start = random_episode_start
        self.cash = cash
        self.max_slippage_percent = max_slippage_percent
        self.lot_size = lot_size
        self.leverage = leverage
        self.pip_size = pip_size
        self.pip_spread = pip_spread
        self.compute_position = compute_position
        self.compute_indicators = compute_indicators
        self.compute_reward = compute_reward
        self.verbose = verbose

        self.pf_stats = pf_stats()
        self.convert = convert('USD', self.instruments, self.lot_size, self.leverage)
        self.reward_scheme = reward_scheme(self.number_of_instruments, self.compute_reward)

        self.price_data, self.features_data = self._init_market_data()

        if self.verbose:
            print('price data has NaNs: {}, has infs: {}'.format(np.any(np.isnan(self.price_data)),
                                                                 np.any(np.isinf(self.price_data))))
            print('price data shape: {}'.format(np.shape(self.price_data)))

            print('features data has NaNs: {}, has infs: {}'.format(np.any(np.isnan(self.features_data)),
                                                                    np.any(np.isinf(self.features_data))))
            print('features data shape: {}'.format(np.shape(self.features_data)))

        assert np.sum(np.isnan(self.price_data)) == 0
        assert np.sum(np.isnan(self.features_data)) == 0

        self.action_space = spaces.Box(-1, 1, shape=(self.number_of_instruments + 1,), dtype=np.float16)
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(self.number_of_instruments * (self.features_data.shape[-1] + 1),),
                                            dtype=np.float16)

        print(f'Action space: {self.action_space}')
        print(f'Observation_space: {self.observation_space}')

        self.last_suggested_weights = np.concatenate(
            (np.zeros(len(self.instruments)), [1.]))  # starting with cash only
        self.last_suggested_positions = np.concatenate(
            (np.zeros(len(self.instruments)), [1.]))  # starting with cash only
        self.last_portfolio_returns = []
        self.last_portfolio_log_returns = []

        self.last_portfolio_value = self.cash
        self.last_infos = []
        self.last_rewards = []

        self.action_sample = self.action_space.sample()

        self.observation = self.reset()

    def seed(self, seed=None):

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        self.suggested_weights = []
        self.actual_weights = []
        self.portfolio_positions = []
        self.portfolio_values = []
        self.portfolio_value = []
        self.portfolio_returns = []
        self.portfolio_log_returns = []

        self.rewards = []
        self.infos = []

        if self.random_episode_start:
            self.current_episode_step = random.randint(self.lookback, self.features_data.shape[1] - 1)
        else:
            self.current_episode_step = self.lookback

        self.current_step = 1
        self.current_suggested_weights = np.concatenate(
            (np.zeros(self.number_of_instruments), [1.]))  # starting with cash only
        self.current_actual_weights = np.concatenate(
            (np.zeros(self.number_of_instruments), [1.]))  # starting with cash only
        self.current_portfolio_positions = np.zeros(self.number_of_instruments)
        self.current_pnl_in_account_currency = 0
        self.current_portfolio_values = np.concatenate(
            (np.zeros(self.number_of_instruments), [self.cash]))  # starting with cash only
        self.current_portfolio_value = self.cash
        self.current_portfolio_return = 0
        self.current_portfolio_log_return = 0

        self.current_reward = 0
        self.current_info = {}

        self.suggested_weights.append(self.current_suggested_weights)
        self.actual_weights.append(self.current_actual_weights)
        self.portfolio_positions.append(self.current_portfolio_positions)
        self.portfolio_values.append(self.current_portfolio_values)
        self.portfolio_value.append(self.current_portfolio_value)
        self.portfolio_returns.append(self.current_portfolio_return)
        self.portfolio_log_returns.append(self.current_portfolio_log_return)

        self.rewards.append(self.current_reward)
        self.infos.append(self.current_info)

        self.seed(1)

        return self._get_obs()

    def step(self, action):

        self.current_step += 1

        self.current_suggested_weights = self.get_weights(action)
        self.last_suggested_weights = self.current_suggested_weights
        self.suggested_weights.append(self.current_suggested_weights)

        self.open_prices = self.price_data[:, self.current_episode_step, 0]  # next open price
        self.close_prices = self.price_data[:, self.current_episode_step, 3]  # next close price

        #   add slippage
        if self.max_slippage_percent > 0:
            self.open_prices, self.close_prices = self._add_slippage(self.current_suggested_weights, self.open_prices,
                                                                     self.close_prices)

        try:
            self._rebalance(self.current_suggested_weights, self.open_prices, self.close_prices)
        except Exception as e:
            if self.verbose:
                print(e)
            self.current_reward = math_utils.NEGATIVE_REWARD

        self.last_suggested_positions = self.current_portfolio_positions
        self.last_portfolio_value = self.current_portfolio_value
        self.last_portfolio_returns = self.portfolio_returns

        try:
            self.current_reward = self.reward_scheme.get_reward(self.portfolio_returns, self.portfolio_log_returns)
        except Exception as e:
            if self.verbose:
                print(e)
            self.current_reward = math_utils.NEGATIVE_REWARD

        if np.isnan(self.current_reward) or np.isinf(self.current_reward):
            #   or np.allclose(self.suggested_weights[-1], self.suggested_weights[-2], rtol=1e-02, atol=1e-02, equal_nan=False):
            self.current_reward = math_utils.NEGATIVE_REWARD

        self.rewards.append(self.current_reward)
        self.last_portfolio_value = self.current_portfolio_value

        self.current_info = {
            'current_episode_step': self.current_episode_step,
            'instruments': self.instruments,
            'current_suggested_weights': self.suggested_weights[-1],
            'current_actual_weights': self.actual_weights[-1],
            'current_portfolio_positions': self.portfolio_positions[-1],
            'current_portfolio_values': self.portfolio_values[-1],
            'current_portfolio_value': self.portfolio_value[-1],
            'current_portfolio_returns': self.portfolio_returns[-1],
            'reward': self.rewards[-1],
        }

        self.infos.append(self.current_info)

        if self.verbose:
            pprint(self.infos[-1], width=1)

        self.last_infos = self.infos
        self.last_rewards = self.rewards

        self.current_episode_step += 1

        '''   
        max_drawdown = self._max_drawdown(self.portfolio_returns)
        sharpe = self._sharpe(self.portfolio_returns)
        '''

        self.done = bool(self.current_episode_step == self.features_data.shape[1])

        #   self.done = bool(self.current_episode_step == self.features_data.shape[1] or self.current_portfolio_value < CURRENT_PORTFOLIO_VALUE * self.cash or max_drawdown > MAX_DRAWDOWN or sharpe < SHARPE_RATIO)
        #   self.done = bool(self.current_episode_step == self.features_data.shape[1] or self.current_portfolio_value < 0.9 * self.cash or max_drawdown > 0.1 or sharpe < 2)
        #   self.done = bool(self.current_episode_step == self.features_data.shape[1] or self.current_portfolio_value < 0.75 * self.cash or max_drawdown > 0.25 or sharpe < 1)
        #   self.done = bool(self.current_episode_step == self.features_data.shape[1] or max_drawdown > 0.2)
        #   self.done = bool(self.current_episode_step == self.features_data.shape[1] or self.current_portfolio_value < 0.7 * self.cash
        self.done = bool(self.current_episode_step == self.features_data.shape[1] or self.current_portfolio_value < 0)

        return self._get_obs(), self.rewards[-1], self.done, self.infos[-1]
        #   return self._get_obs(), self.rewards[-1]-np.std(self.rewards), self.done, self.infos[-1]
        #   return self._get_obs(), self.rewards[-1] - statistics.variance(self.rewards), self.done, self.infos[-1]
        #   return self._get_obs(), np.mean(self.rewards) / np.std(self.rewards), self.done, self.infos[-1]

    def get_weights(self, action):

        if self.compute_position == 'long_only':
            weights = np.clip(action, 0, 1)
        elif self.compute_position == 'short_only':
            weights = np.clip(action, -1, 0)
        elif self.compute_position == 'long_and_short':
            weights = np.clip(action, -1, 1)

        if self.verbose:
            print(f'action: {action}')
            print(f'weights: {weights}')

        weights /= (math_utils.my_sum_abs(weights) + math_utils.EPS)
        weights[-1] += np.clip(1 - math_utils.my_sum_abs(weights), 0,
                               1)  # if weights are all zeros we normalise to [0,0..1]
        weights[-1] = math_utils.my_sum_abs(weights[-1])

        if np.all(np.isnan(weights)):
            weights[:] = 0
            weights[-1] = 1

        #   np.testing.assert_almost_equal(my_math.my_sum_abs(weights), 1.0, 3, 'absolute weights should sum to 1. weights=%s' %weights)

        if self.compute_position == 'long_only':
            assert ((weights >= 0) * (
                    weights <= 1)).all(), 'all weights values should be between 0 and 1. Not %s' % weights
        elif self.compute_position == 'short_only':
            assert ((weights[0:-1] >= -1) * (weights[0:-1] <= 0) * (weights[-1] >= 0) * (
                    weights[-1] <= 1)).all(), 'all weights values should be between -1 and 0. Not %s' % weights
        elif self.compute_position == 'long_only':
            assert ((weights[0:-1] >= -1) * (weights[0:-1] <= 1) * (weights[-1] >= 0) * (
                    weights[-1] <= 1)).all(), 'all weights values should be between -1 and 1. Not %s' % weights

        return weights

    def _get_obs(self):

        data = self.features_data[:, self.current_episode_step - self.lookback:self.current_episode_step, :]
        state = (np.divide(np.subtract(data, np.mean(data, axis=1, keepdims=True)),
                           (np.std(data, axis=1, keepdims=True) + math_utils.EPS)))[:, -1, :]
        normalized_state = (np.concatenate((state, self.current_actual_weights[:-1][:, None]), axis=1))
        #   self.normalized_state_flatten = np.nan_to_num(normalized_state).flatten()
        self.normalized_state_flatten = normalized_state.flatten()
        self.normalized_state_shape = np.shape(normalized_state.flatten())

        '''
        dict = {'normalized_state_flatten': self.normalized_state_flatten}
        with open('Debug2.txt', 'w') as file:
            file.write(json.dumps(dict))
        '''

        if self.verbose:
            print(f'Normalized state flatten: {self.normalized_state_flatten}')
            print(f'Normalized state flatten shape: {self.normalized_state_shape}')
            print(f'Observation Space: {self.observation_space}')

        return self.normalized_state_flatten

    def _add_spread(self, weights, open_prices, close_prices):

        i = 0
        for pip_size, pip_spread in zip(self.pip_size, self.pip_spread):
            if weights[i] > 0:
                open_prices[i] = open_prices[i] + pip_size * pip_spread
                close_prices[i] = close_prices[i] - pip_size * pip_spread
            elif weights[i] < 0:
                open_prices[i] = open_prices[i] - pip_size * pip_spread
                close_prices[i] = close_prices[i] + pip_size * pip_spread
            i += 1

        return open_prices, close_prices

    def _add_slippage(self, weights, open_prices, close_prices):

        i = 0
        for _ in self.instruments:
            if weights[i] > 0:
                slippage = np.random.uniform(0, self.max_slippage_percent / 100, 1)
                open_prices[i] = open_prices[i] * (1 + slippage)
                slippage = np.random.uniform(0, self.max_slippage_percent / 100, 1)
                close_prices[i] = close_prices[i] * (1 - slippage)
            elif weights[i] < 0:
                slippage = np.random.uniform(0, self.max_slippage_percent / 100, 1)
                open_prices[i] = open_prices[i] * (1 - slippage)
                slippage = np.random.uniform(0, self.max_slippage_percent / 100, 1)
                close_prices[i] = close_prices[i] * (1 + slippage)
            i += 1

        return open_prices, close_prices

    def _rebalance(self, weights, open_prices, close_prices):

        portfolio_value_begin = self.current_portfolio_value

        #   Rebalance

        self.current_portfolio_values = np.multiply(self.current_portfolio_value, weights)
        current_margins_per_lot = self.convert.calculate_margin_in_account_currency(open_prices)
        self.current_portfolio_positions = np.divide(self.current_portfolio_values[:-1], current_margins_per_lot)

        #   PNL

        prices_move = np.subtract(close_prices, open_prices)
        pips_move = np.divide(prices_move, self.pip_size)

        pips_move_net = []

        for cpp, pm, ps in zip(self.current_portfolio_positions, pips_move, self.pip_spread):
            if cpp > 0:
                pips_move_net.append(pm - ps)
            elif cpp < 0:
                pips_move_net.append(-(pm - ps))
            elif cpp == 0:
                pips_move_net.append(0)

        if self.verbose:
            print(f'pips_move_net: {pips_move_net}')
            print(f'current_portfolio_positions: {self.current_portfolio_positions}')

        total_pips = np.multiply(pips_move_net, self.current_portfolio_positions)
        close_pip_values_in_account_currency = self.convert.calculate_pip_value_in_account_currency(close_prices)

        self.current_pnl_in_account_currency = np.multiply(total_pips, close_pip_values_in_account_currency)
        self.current_portfolio_values[:-1] += self.current_pnl_in_account_currency
        self.current_portfolio_value = math_utils.my_sum_abs(self.current_portfolio_values)
        self.current_actual_weights = np.divide(self.current_portfolio_values, self.current_portfolio_value)

        portfolio_value_end = self.current_portfolio_value

        self.actual_weights.append(self.current_actual_weights)
        self.portfolio_positions.append(self.current_portfolio_positions)
        self.portfolio_values.append(self.current_portfolio_values)
        self.portfolio_value.append(self.current_portfolio_value)

        self.current_portfolio_return = portfolio_value_end / portfolio_value_begin - 1
        self.current_portfolio_log_return = np.log(portfolio_value_end) - np.log(portfolio_value_begin)

        self.portfolio_returns.append(self.current_portfolio_return)
        self.portfolio_log_returns.append(self.current_portfolio_log_return)

    def _init_market_data(self):

        new_data = np.zeros((0, 0, 0), dtype=np.float32)

        for i in range(self.data.shape[0]):
            security = pd.DataFrame(self.data[i, :, :]).fillna(method='ffill').fillna(method='bfill')
            security.columns = ['Open', 'High', 'Low', 'Close']

            self.technical_features = technical_features(security=security.astype(float), open_name='Open',
                                                         high_name='High', low_name='Low', close_name='Close')

            if self.compute_indicators == 'prices':
                features_data = np.asarray(self.technical_features.get_indicators_prices())
            elif self.compute_indicators == 'returns':
                features_data = np.asarray(self.technical_features.get_indicators_returns())
            elif self.compute_indicators == 'log_returns':
                features_data = np.asarray(self.technical_features.get_indicators_log_returns())
            elif self.compute_indicators == 'returns_hlc':
                features_data = np.asarray(self.technical_features.get_indicators_returns_hlc())
            elif self.compute_indicators == 'log_returns_hlc':
                features_data = np.asarray(self.technical_features.get_indicators_log_returns_hlc())
            elif self.compute_indicators == 'patterns':
                features_data = np.asarray(self.technical_features.get_indicators_patterns())
            elif self.compute_indicators == 'returns_patterns_volatility':
                features_data = np.asarray(self.technical_features.get_indicators_returns_patterns_volatility())
            elif self.compute_indicators == 'momentum_simple':
                features_data = np.asarray(self.technical_features.get_indicators_momentum('simple'))
            elif self.compute_indicators == 'momentum_multi':
                features_data = np.asarray(self.technical_features.get_indicators_momentum('multi'))
            elif self.compute_indicators == 'all':
                features_data = np.asarray(self.technical_features.get_indicators_all())
            elif self.compute_indicators == 'misc':
                features_data = np.asarray(self.technical_features.get_indicators_misc())
            elif self.compute_indicators == 'random':
                features_data = np.asarray(self.technical_features.get_indicators_random())

            new_data = np.resize(new_data, (new_data.shape[0] + 1, features_data.shape[0], features_data.shape[1]))
            new_data[i] = features_data

            price_data = new_data[:, :, :4]
            features_data = new_data[:, :, 4:]

        return price_data, features_data
