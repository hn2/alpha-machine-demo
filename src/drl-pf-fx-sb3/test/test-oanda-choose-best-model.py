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
from collections import deque
from datetime import timedelta

from findrl.forex_utils import get_forex_7, get_forex_12, get_forex_18, get_forex_28
from findrl.model_utils import parse_model_name
from findrl.run_utils import run_choose_best_model

#   cd util_lib
#   python -m tensorboard.main --logdir=.
#   tensorboard --logdir=.

DELIMITER = '-'
ENV_VERBOSE = True

CHOOSE_BEST_MODELS_DIR = r'E:\alpha-machine\models\forex\oanda\daily\cbm'
#   CHOOSE_BEST_MODEL_NAME = 'fx_sb3_leverage_20_train_with_callback_with_random_episode_start_noise_none-7-100100-1000-100-2-oanda-daily-on_algo.ppo-comp_pos.long_and_short-comp_ind.all-comp_rew.[log_returns]-05f1327f'
CHOOSE_BEST_MODEL_NAME = 'fx_sb3_leverage_20_train_with_callback_with_random_episode_start_noise_none-7-100100-1000-100-2-oanda-daily-on_algo.ppo-comp_pos.long_and_short-comp_ind.all-comp_rew.[log_returns]-34075aee'
MODELS_DIR = r'E:\alpha-machine\models\forex\oanda\daily\prod'
FILES_LOOKBACK_HOURS = 2400
INCLUDE_PATTERNS = ['ppo']


class main(QCAlgorithm):

    def Initialize(self):
        self.number_of_instruments, self.market_name, self.resolution_name, self.env_lookback_period, self.spread, self.online_algorithm, self.compute_position, self.compute_indicators, self.compute_reward = parse_model_name(
            CHOOSE_BEST_MODEL_NAME, DELIMITER)

        self.Debug(self.number_of_instruments)
        self.Debug(self.market_name)
        self.Debug(self.resolution_name)
        self.Debug(self.env_lookback_period)
        self.Debug(self.spread)
        self.Debug(self.online_algorithm)
        self.Debug(self.compute_position)
        self.Debug(self.compute_indicators)
        self.Debug(self.compute_reward)

        if int(self.number_of_instruments) == 7:
            self.instruments, self.pip_size, self.pip_spread = get_forex_7(int(self.spread))
        elif int(self.number_of_instruments) == 12:
            self.instruments, self.pip_size, self.pip_spread = get_forex_12(int(self.spread))
        elif int(self.number_of_instruments) == 18:
            self.instruments, self.pip_size, self.pip_spread = get_forex_18(int(self.spread))
        elif int(self.number_of_instruments) == 28:
            self.instruments, self.pip_size, self.pip_spread = get_forex_28(int(self.spread))

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

        # self.SetStartDate(2021, 8, 15)
        # self.SetEndDate(2021, 11, 23)

        self.SetStartDate(2015, 8, 15)
        self.SetEndDate(2016, 11, 23)

        self.Debug("--------------")
        self.Debug(self.instruments)
        self.Debug(multiprocessing.get_all_start_methods())
        self.Debug(multiprocessing.cpu_count())
        self.Debug("--------------")

        self.lot_size = 'Micro'
        self.leverage = 20.0
        self.max_slippage_percent = 1e-2
        #   self.max_slippage_percent = 0

        self.cash = 1e3
        self.SetCash(self.cash)

        self.trade_lookback_period = self.env_lookback_period + 200

        self.Debug(self.env_lookback_period)
        self.Debug(self.trade_lookback_period)

        '''
        now = datetime.now()
        self.SetStartDate(2021, 8, 14)
        self.SetEndDate(2021, 11, 22)
        '''

        # online model
        self.online_model = None
        self.deterministic = True
        self.verbose = 1

        self.rollingwindow = {}
        symbols = self.Securities.Keys
        history = self.History(symbols, self.trade_lookback_period, self.resolution)
        for symbol in symbols:
            self.rollingwindow[symbol] = QuoteBarData(self, symbol, self.trade_lookback_period, history.loc[symbol],
                                                      self.resolution)

        #   self.Schedule.On(self.DateRules.EveryDay("EURUSD"), self.TimeRules.BeforeMarketClose("EURUSD", 5),
        #                    self.LiquidateAtClose)

    def OnData(self, data):
        '''
        for key in data.Keys:
            self.Log(str(key.Value) + ": " + str(data.Time) + " > " + str(data[key].Value))
        '''
        for _, QuoteBarData in self.rollingwindow.items():
            self.Debug("QuoteBarData")

        self.rebalance()

    def LiquidateAtClose(self):
        self.Liquidate()

    def rebalance(self):
        for symbol, QuoteBarData in self.rollingwindow.items():

            open = np.asarray(QuoteBarData.open)
            if open.shape[0] < self.trade_lookback_period:
                return

        self.Debug('Running model...')

        data = self.prepare_data_trade()

        suggested_weights, suggested_positions = run_choose_best_model(CHOOSE_BEST_MODELS_DIR,
                                                                       CHOOSE_BEST_MODEL_NAME,
                                                                       MODELS_DIR,
                                                                       FILES_LOOKBACK_HOURS,
                                                                       INCLUDE_PATTERNS,
                                                                       DELIMITER,
                                                                       self.deterministic,
                                                                       data,
                                                                       self.instruments,
                                                                       self.env_lookback_period,
                                                                       False,
                                                                       self.cash,
                                                                       self.max_slippage_percent,
                                                                       self.lot_size,
                                                                       self.leverage,
                                                                       self.pip_size,
                                                                       self.pip_spread,
                                                                       self.compute_position,
                                                                       self.compute_indicators,
                                                                       self.compute_reward,  # returns log_returns
                                                                       ENV_VERBOSE)

        self.Debug(self.Time)
        self.Debug(self.UtcTime)
        self.Debug(f"Date:{self.Time.date()}")
        self.Debug(self.instruments)
        self.Debug(f'Suggested weights: {suggested_weights}')
        self.Debug(f'Suggested positions: {suggested_positions}')
        self.Debug(f"Before rebalance:{self.Portfolio.TotalPortfolioValue}")

        #   self.Liquidate()

        for i, instrument in enumerate(self.instruments):
            self.SetHoldings(instrument, self.leverage * suggested_weights[i])

        self.Debug(f"After rebalance:{self.Portfolio.TotalPortfolioValue}")
        self.Debug("================")

    def prepare_data_trade(self):
        data = np.empty(shape=(len(self.instruments), 0, 4), dtype=np.float)

        i = 0

        for instrument in self.instruments:
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
