#   https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/risk-management/supported-models

from clr import AddReference

AddReference("System")
AddReference("QuantConnect.Algorithm")
AddReference("QuantConnect.Indicators")
AddReference("QuantConnect.Common")

from QuantConnect import *
from QuantConnect.Data import *
from QuantConnect.Algorithm import *
from QuantConnect.Brokerages import *
from QuantConnect.Data.Consolidators import *

import numpy as np
import multiprocessing
from os.path import join as path_join
from collections import deque
from datetime import timedelta
from distutils.util import strtobool

from findrl.model_utils import parse_model_name
from findrl.math_utils import math_utils
from findrl.model_utils import get_models_and_stats_lists_from_position
from findrl.file_utils import get_models, load_dict_from_pickle
from findrl.forex_utils import get_forex_28
from findrl.run_utils import run_model_make_env_sb3
from findrl.general_utils import convert_lists_to_dict


#   STATS_FILE = r'C:\alpha-machine\src\drl-pf-fx-sb3\stats\statistics-prod-Oanda-OandaBrokerage-daily-21-12-2021-0-100-.csv'

# columns = ['Timestamp', 'Total Trades', 'Average Win', 'Average Loss', 'Compounding Annual Return',
#            'Drawdown', 'Expectancy', 'Net Profit', 'Sharpe Ratio',
#            'Probabilistic Sharpe Ratio', 'Loss Rate', 'Win Rate', 'Profit-Loss Ratio',
#            'Alpha', 'Beta', 'Annual Standard Deviation', 'Annual Variance',
#            'Information Ratio', 'Tracking Error', 'Treynor Ratio', 'Total Fees',
#            'Fitness Score', 'Kelly Criterion Estimate', 'Kelly Criterion Probability Value',
#            'Sortino Ratio', 'Return Over Maximum Drawdown', 'Portfolio Turnover',
#            'Total Insights Generated', 'Total Insights Closed', 'Total Insights Analysis Completed',
#            'Long Insight Count', 'Short Insight Count', 'Long/Short Ratio', 'Estimated Monthly Alpha Value',
#            'Total Accumulated Estimated Alpha Value', 'Mean Population Estimated Insight Value',
#            'Mean Population Direction', 'Mean Population Magnitude', 'Rolling Averaged Population Direction',
#            'Rolling Averaged Population Magnitude', 'OrderListHash', 'Model Name']

class main(QCAlgorithm):

    def Initialize(self):
        if STATS_FILE is None:
            self.model_names = get_models(MODELS_DIR, FILES_LOOKBACK_HOURS, INCLUDE_PATTERNS)
        else:
            self.params = {'algos': ['None'], 'sort_columns': 'Expectancy',
                           'stat_column': 'Expectancy', 'position': 1,
                           'number': NUM_OF_MODELS}
            self.model_names, self.weights = get_models_and_stats_lists_from_position(stats_file=STATS_FILE,
                                                                                      params=self.params)

        self.Debug(self.model_names)
        #   self.Debug(self.weights)

        self.number_of_instruments, self.total_timesteps, self.market_name, self.resolution_name, self.env_lookback_period, self.spread, self.online_algorithm, self.compute_position, self.compute_indicators, self.compute_reward, self.meta_rl = parse_model_name(
            self.model_names[0], DELIMITER)

        self.meta_rl = strtobool(self.meta_rl)

        self.Debug(self.number_of_instruments)
        self.Debug(self.total_timesteps)
        self.Debug(self.market_name)
        self.Debug(self.resolution_name)
        self.Debug(self.env_lookback_period)
        self.Debug(self.spread)
        self.Debug(self.online_algorithm)
        self.Debug(self.compute_position)
        self.Debug(self.compute_indicators)
        self.Debug(self.meta_rl)
        self.Debug(self.compute_reward)

        # env_attributes = load_dict_from_pickle(path_join(*[MODELS_DIR, self.model_names[0], 'online', 'env.pkl']))
        # self.account_currency = env_attributes['account_currency']
        # self.instruments = env_attributes['instruments']
        # self.pip_size = env_attributes['pip_size']
        # self.pip_spread = env_attributes['pip_spread']

        self.instruments, self.pip_size, self.pip_spread = get_forex_28(2)

        if self.market_name == 'oanda':
            self.market = Market.Oanda
        elif self.market_name == 'fxcm':
            self.market = Market.FXCM

        if self.resolution_name == 'daily' or self.resolution_name == 'day':
            self.resolution = Resolution.Daily
        elif self.resolution_name == 'hour':
            self.resolution = Resolution.Hour

        self.symbols = [self.AddForex(instrument, self.resolution, self.market).Symbol for instrument in
                        self.instruments]
        self.SetBrokerageModel(BrokerageName.OandaBrokerage)

        self.SetStartDate(START_DATE[0], START_DATE[1], START_DATE[2])
        self.SetEndDate(END_DATE[0], END_DATE[1], END_DATE[2])

        self.Debug(f'Start date: {self.StartDate}')
        self.Debug(f'End date: {self.EndDate}')

        self.Debug("--------------")
        self.Debug(self.instruments)
        self.Debug(multiprocessing.get_all_start_methods())
        self.Debug(multiprocessing.cpu_count())
        self.Debug("--------------")

        self.lot_size = LOT_SIZE
        self.leverage = LEVERAGE
        self.max_slippage_percent = MAX_SLIPPAGE_PERCENT
        #   self.max_slippage_percent = 0

        self.cash = CASH
        self.SetCash(self.cash)

        #   self.AddRiskManagement(MaximumDrawdownPercentPortfolio(maximumDrawdownPercent=0.5))
        #   self.AddRiskManagement(MaximumDrawdownPercentPerSecurity(maximumDrawdownPercent=0.5))
        #   self.AddRiskManagement(MaximumUnrealizedProfitPercentPerSecurity(maximumUnrealizedProfitPercent=0.1))
        #   self.AddRiskManagement(TrailingStopRiskManagementModel(maximumDrawdownPercent=0.5))

        self.trade_lookback_period = self.env_lookback_period + 100

        self.Debug(self.env_lookback_period)
        self.Debug(self.trade_lookback_period)

        '''
        now = datetime.now()
        self.SetStartDate(2021, 10, 14)
        self.SetEndDate(2022, 1, 22)
        '''

        # online model
        self.online_model = None
        self.deterministic = True
        self.env_verbose = False

        self.rollingwindow = {}
        symbols = self.Securities.Keys
        history = self.History(symbols, self.trade_lookback_period, self.resolution)
        for symbol in symbols:
            self.rollingwindow[symbol] = QuoteBarData(self, symbol, self.trade_lookback_period, history.loc[symbol],
                                                      self.resolution)

        symbols_in_portfolio = [x.Value for x in self.Portfolio.Keys]

        if self.resolution == Resolution.Daily:
            self.Schedule.On(self.DateRules.EveryDay(symbols_in_portfolio[0]), self.TimeRules.At(0, 0), self.rebalance)
        elif self.resolution == Resolution.Hour:
            self.Schedule.On(self.DateRules.EveryDay(symbols_in_portfolio[0]),
                             self.TimeRules.Every(timedelta(minutes=60)), self.rebalance)
        elif self.resolution == Resolution.Minute:
            self.Schedule.On(self.DateRules.EveryDay(symbols_in_portfolio[0]),
                             self.TimeRules.Every(timedelta(minutes=1)), self.rebalance)

    def OnData(self, data):
        pass

    def rebalance(self):
        for symbol, QuoteBarData in self.rollingwindow.items():

            open = np.asarray(QuoteBarData.open)
            if open.shape[0] < self.trade_lookback_period:
                return

        self.Debug('Running model...')

        v_suggested_weights_all = np.zeros(shape=len(self.instruments) + 1, dtype=np.float)
        v_suggested_weights_dict_all = convert_lists_to_dict(self.instruments, v_suggested_weights_all)
        number_of_models = len(self.model_names)

        # self.Debug(self.instruments)
        # self.Debug(v_suggested_weights_all)
        # self.Debug(v_suggested_weights_dict_all)

        for i, model_name in enumerate(self.model_names):
            number_of_instruments, total_timesteps, market_name, resolution_name, env_lookback_period, spread, online_algorithm, compute_position, compute_indicators, compute_reward, meta_rl = parse_model_name(
                model_name, DELIMITER)

            meta_rl = strtobool(meta_rl)

            env_attributes = load_dict_from_pickle(path_join(*[MODELS_DIR, model_name, 'online', 'env.pkl']))
            account_currency = env_attributes.account_currency
            self.current_instruments = env_attributes.instruments
            pip_size = env_attributes.pip_size
            pip_spread = env_attributes.pip_spread

            # self.Debug(self.current_instruments)
            # self.Debug(pip_size)
            # self.Debug(pip_spread)

            data = self.prepare_data_trade()

            v_suggested_weights, v_suggested_positions = run_model_make_env_sb3(MODELS_DIR,
                                                                                model_name,
                                                                                DELIMITER,
                                                                                self.deterministic,
                                                                                data,
                                                                                account_currency,
                                                                                self.current_instruments,
                                                                                self.env_lookback_period,
                                                                                False,
                                                                                self.cash,
                                                                                self.max_slippage_percent,
                                                                                self.lot_size,
                                                                                self.leverage,
                                                                                pip_size,
                                                                                pip_spread,
                                                                                self.compute_position,
                                                                                self.compute_indicators,
                                                                                self.compute_reward,
                                                                                meta_rl,
                                                                                self.env_verbose)

            v_suggested_weights_dict = convert_lists_to_dict(self.current_instruments, v_suggested_weights)

            for key, value in v_suggested_weights_dict.items():
                v_suggested_weights_dict_all[key] += v_suggested_weights_dict.get(key)

        #   v_weights = np.average(v_suggested_weights_all, axis=0, weights=self.weights)
        v_weights = list(v_suggested_weights_dict_all.values())
        v_weights = [float(x) / number_of_models for x in v_weights]
        v_normelized_weights = math_utils.get_normelize_weights(v_weights)

        self.Debug(self.Time)
        self.Debug(self.UtcTime)
        self.Debug(f"Date:{self.Time.date()}")
        self.Debug(self.instruments)
        self.Debug(f'Normelized weights: {v_normelized_weights}')
        self.Debug(f"Before rebalance:{self.Portfolio.TotalPortfolioValue}")

        #   self.Liquidate()

        for i, instrument in enumerate(self.instruments):
            self.SetHoldings(instrument, self.leverage * v_normelized_weights[i])

        self.Debug(f"After rebalance:{self.Portfolio.TotalPortfolioValue}")

    def prepare_data_trade(self):
        data = np.empty(shape=(len(self.current_instruments), 0, 4), dtype=np.float)

        i = 0

        for instrument in self.current_instruments:
            history = self.History(self.Symbol(instrument), self.trade_lookback_period, self.resolution)

            df_data = history[['open', 'high', 'low', 'close']]
            np_data = np.array(df_data)

            data = np.resize(data, (data.shape[0], np_data.shape[0], data.shape[2]))

            data[i] = np_data

            i += 1

        return data


class QuoteBarData:

    def __init__(self, algorithm, symbol, lookback, history, resolution):
        self.open = deque(maxlen=lookback)
        self.high = deque(maxlen=lookback)
        self.low = deque(maxlen=lookback)
        self.close = deque(maxlen=lookback)
        self.Debug = algorithm.Debug

        self.time = algorithm.Time

        if resolution == Resolution.Daily:
            consolidator = QuoteBarConsolidator(timedelta(days=1))
        elif resolution == Resolution.Hour:
            consolidator = QuoteBarConsolidator(timedelta(hours=1))
        elif resolution == Resolution.Minute:
            consolidator = QuoteBarConsolidator(timedelta(minutes=1))

        algorithm.SubscriptionManager.AddConsolidator(symbol, consolidator)
        consolidator.DataConsolidated += self.OnDataConsolidated

        for time, row in history.iterrows():
            self.time = time
            self.Update(time, row.open, row.high, row.low, row.close)

    def Update(self, time, o, h, l, c):
        self.time = time
        self.open.append(o)
        self.high.append(h)
        self.low.append(l)
        self.close.append(c)

    def OnDataConsolidated(self, sender, bar):
        self.Update(bar.EndTime, bar.Open, bar.High, bar.Low, bar.Close)
